from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app import rag
from app.config import Settings
from app import store


class RecordingQuery:
    def __init__(self):
        self.call = None

    def near_vector(self, **kwargs):
        self.call = ("dense", kwargs)
        return self.call

    def hybrid(self, **kwargs):
        self.call = ("hybrid", kwargs)
        return self.call


class RecordingCollection:
    def __init__(self):
        self.query = RecordingQuery()


def test_dense_retrieval_uses_near_vector_and_preserves_filters():
    collection = RecordingCollection()
    filters = rag.build_filters({"2606.05387": 2}, "2606.05387", 1)

    result = rag.retrieve_chunks(
        collection, "question", [0.1, 0.2], filters, mode="dense", limit=5
    )

    assert result[0] == "dense"
    assert result[1] == {
        "near_vector": [0.1, 0.2],
        "limit": 5,
        "filters": filters,
    }
    assert [
        (item.target, item.value)
        for item in filters.filters
    ] == [("arxiv_id", "2606.05387"), ("version", 1)]


def test_hybrid_retrieval_uses_configured_alpha_and_version_filter():
    collection = RecordingCollection()
    filters = rag.build_filters({"2606.05387": 2}, "2606.05387", 1)

    result = rag.retrieve_chunks(
        collection, "question", [0.1, 0.2], filters, mode="hybrid", alpha=0.25
    )

    assert result[0] == "hybrid"
    assert result[1] == {
        "query": "question",
        "vector": [0.1, 0.2],
        "alpha": 0.25,
        "limit": Settings(_env_file=None).retrieval_top_k,
        "filters": filters,
    }
    assert [(item.target, item.value) for item in filters.filters] == [
        ("arxiv_id", "2606.05387"),
        ("version", 1),
    ]


def test_retrieval_configuration_validation():
    assert Settings(_env_file=None, retrieval_mode="dense").retrieval_mode == "dense"
    assert Settings(_env_file=None, hybrid_alpha=0.0).hybrid_alpha == 0.0
    assert Settings(_env_file=None, hybrid_alpha=1.0).hybrid_alpha == 1.0
    with pytest.raises(ValidationError):
        Settings(_env_file=None, retrieval_mode="bm25")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, hybrid_alpha=-0.01)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, hybrid_alpha=1.01)


def test_retrieval_rejects_invalid_mode_and_alpha():
    collection = RecordingCollection()
    with pytest.raises(ValueError, match="Unsupported retrieval mode"):
        rag.retrieve_chunks(collection, "question", [], None, mode="other")
    with pytest.raises(ValueError, match="alpha"):
        rag.retrieve_chunks(collection, "question", [], None, mode="hybrid", alpha=1.1)


def test_latest_filter_uses_latest_indexed_version():
    filters = rag.build_filters(
        {"2605.18821": 1, "2606.05387": 2, "2607.00365": 1}
    )

    assert filters.target == "version_key"
    assert set(filters.value) == {
        "2605.18821v1",
        "2606.05387v2",
        "2607.00365v1",
    }


def test_new_paper_chunk_schema_indexes_text_for_bm25():
    class Collections:
        def __init__(self):
            self.created = {}

        def exists(self, name):
            return False

        def create(self, name, **kwargs):
            self.created[name] = kwargs

    client = SimpleNamespace(collections=Collections())

    store.setup(client)

    text_property = next(
        prop for prop in client.collections.created["PaperChunk"]["properties"]
        if prop.name == "text"
    )
    assert text_property.model_dump(by_alias=True)["index_searchable"] is True


def test_comparison_intent_bypasses_normal_retrieval(monkeypatch):
    expected = {"answer": "comparison result", "sources": []}
    monkeypatch.setattr(rag, "answer_version_comparison", lambda *args, **kwargs: expected)
    monkeypatch.setattr(rag, "encoder", lambda: pytest.fail("encoder must not run"))
    monkeypatch.setattr(rag, "connect", lambda: pytest.fail("Weaviate retrieval must not run"))

    result = rag.answer("What changed between versions 1 and 2 of arXiv:2606.05387?")

    assert result == expected


def test_citation_validation_still_selects_valid_sources():
    sources = [{"label": "S1"}, {"label": "S2"}]

    assert rag.validate_citations("Supported by [S2].", sources) == [sources[1]]
    with pytest.raises(ValueError, match="Invalid citations"):
        rag.validate_citations("Unsupported label [S3].", sources)


def test_evaluation_metrics_require_labels_and_respect_judgment_completeness():
    from scripts.evaluate_retrieval import _metrics

    unlabeled = [{
        "relevant_chunk_ids": None,
        "judgments_complete": False,
        "retrieved_chunk_ids": ["chunk-a"],
    }]
    assert _metrics(unlabeled, 5) == {
        "recall_at_k": None,
        "hit_rate_at_k": None,
        "mrr": None,
        "precision_at_k": None,
        "relevant_queries": 0,
        "complete_judgment_queries": 0,
        "recall_queries_with_relevant_labels": 0,
        "unjudged_queries": 1,
    }

    labeled = [{
        "relevant_chunk_ids": ["chunk-a", "chunk-b"],
        "judgments_complete": True,
        "retrieved_chunk_ids": ["chunk-a", "chunk-c"],
    }]
    metrics = _metrics(labeled, 2)
    assert metrics["recall_at_k"] == 0.5
    assert metrics["hit_rate_at_k"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["precision_at_k"] == 0.5


def test_pooled_judgments_apply_only_reviewed_candidate_labels(tmp_path):
    import json

    from scripts.evaluate_retrieval import _load_dataset

    dataset_path = tmp_path / "queries.json"
    dataset_path.write_text(json.dumps([
        {"id": "q1", "question": "Question", "relevant_chunk_ids": None},
        {"id": "q2", "question": "Another question", "relevant_chunk_ids": None},
    ]), encoding="utf-8")
    judgments_path = tmp_path / "judgments.json"
    judgments_path.write_text(json.dumps({
        "queries": [
            {
                "id": "q1",
                "relevant_chunk_ids": ["chunk-a"],
                "candidate_pool_reviewed": True,
                "candidates": [{"chunk_id": "chunk-a"}],
            },
            {
                "id": "q2",
                "relevant_chunk_ids": [],
                "candidate_pool_reviewed": False,
                "candidates": [{"chunk_id": "chunk-b"}],
            },
        ]
    }), encoding="utf-8")

    queries = _load_dataset(dataset_path, judgments_path)

    assert queries[0]["relevant_chunk_ids"] == ["chunk-a"]
    assert queries[0]["judgments_complete"] is True
    assert queries[1]["relevant_chunk_ids"] is None
    assert queries[1]["judgments_complete"] is False


def test_pooled_judgments_reject_chunk_ids_outside_candidate_pool(tmp_path):
    import json

    from scripts.evaluate_retrieval import _load_dataset

    dataset_path = tmp_path / "queries.json"
    dataset_path.write_text(json.dumps([
        {"id": "q1", "question": "Question"},
    ]), encoding="utf-8")
    judgments_path = tmp_path / "judgments.json"
    judgments_path.write_text(json.dumps({
        "queries": [{
            "id": "q1",
            "relevant_chunk_ids": ["not-in-pool"],
            "candidate_pool_reviewed": True,
            "candidates": [{"chunk_id": "chunk-a"}],
        }]
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="outside its candidate pool"):
        _load_dataset(dataset_path, judgments_path)


def test_judgment_export_pools_candidate_chunks_across_modes(tmp_path):
    import json

    from scripts.prepare_retrieval_judgments import build_template

    dataset_path = tmp_path / "queries.json"
    dataset_path.write_text(json.dumps([
        {
            "id": "q1",
            "question": "Question",
            "category": "test",
            "source_basis": "Test passage",
        }
    ]), encoding="utf-8")
    results_path = tmp_path / "results.json"
    item = {
        "chunk_id": "chunk-a",
        "arxiv_id": "2606.05387",
        "version": 1,
        "page": 2,
        "section": "Intro",
        "text": "Relevant passage",
    }
    results_path.write_text(json.dumps({
        "results": [
            {
                "mode": "dense",
                "alpha": None,
                "queries": [{"id": "q1", "results": [item]}],
            },
            {
                "mode": "hybrid",
                "alpha": 0.5,
                "queries": [{"id": "q1", "results": [item]}],
            },
        ]
    }), encoding="utf-8")

    template = build_template(results_path, dataset_path)
    candidates = template["queries"][0]["candidates"]

    assert len(candidates) == 1
    assert candidates[0]["chunk_id"] == "chunk-a"
    assert [entry["mode"] for entry in candidates[0]["retrieved_by"]] == [
        "dense",
        "hybrid",
    ]
