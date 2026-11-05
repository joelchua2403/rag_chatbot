"""Compare dense and hybrid retrieval against a manually labeled query set."""

import argparse
import json
import random
import statistics
import time
from pathlib import Path

from app.config import settings
from app.rag import build_filters, encoder, get_latest_versions, retrieve_chunks
from app.store import connect

CONFIGURATIONS = (
    ("dense", None),
    ("hybrid", 0.25),
    ("hybrid", 0.50),
    ("hybrid", 0.75),
)


def _metrics(query_results, k):
    judged = []
    complete = []
    unjudged = 0
    for query_result in query_results:
        labels = query_result["relevant_chunk_ids"]
        if labels is None:
            unjudged += 1
            continue
        relevant = set(labels)
        retrieved = query_result["retrieved_chunk_ids"][:k]
        judged.append((relevant, retrieved))
        if query_result["judgments_complete"]:
            complete.append((relevant, retrieved))

    recall = [
        len(relevant.intersection(retrieved)) / len(relevant)
        for relevant, retrieved in judged
        if relevant
    ]
    hit_rate = [
        float(bool(relevant.intersection(retrieved)))
        for relevant, retrieved in judged
    ]
    reciprocal_rank = []
    for relevant, retrieved in judged:
        ranks = [index for index, chunk_id in enumerate(retrieved, start=1) if chunk_id in relevant]
        reciprocal_rank.append(1.0 / ranks[0] if ranks else 0.0)

    return {
        "recall_at_k": sum(recall) / len(recall) if recall else None,
        "hit_rate_at_k": sum(hit_rate) / len(hit_rate) if hit_rate else None,
        "mrr": sum(reciprocal_rank) / len(reciprocal_rank) if reciprocal_rank else None,
        "precision_at_k": (
            sum(len(relevant.intersection(retrieved)) / k for relevant, retrieved in complete)
            / len(complete)
            if complete and k > 0
            else None
        ),
        "relevant_queries": len(judged),
        "complete_judgment_queries": len(complete),
        "recall_queries_with_relevant_labels": sum(
            bool(relevant) for relevant, _ in judged
        ),
        "unjudged_queries": unjudged,
    }


def _load_dataset(path, judgments_path=None):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("Evaluation dataset must be a non-empty JSON list.")
    seen_ids = set()
    for item in data:
        query_id = item.get("id")
        if not query_id or query_id in seen_ids:
            raise ValueError("Each evaluation item must have a unique, non-empty id.")
        seen_ids.add(query_id)
        if item.get("version") is not None and item.get("arxiv_id") is None:
            raise ValueError(f"Query {query_id} specifies a version without an arXiv ID.")
        relevant_ids = item.get("relevant_chunk_ids")
        if relevant_ids is not None and not isinstance(relevant_ids, list):
            raise ValueError(f"Query {query_id} relevant_chunk_ids must be a list or null.")
    if judgments_path is not None:
        judgment_data = json.loads(Path(judgments_path).read_text(encoding="utf-8"))
        judgments = judgment_data.get("queries")
        if not isinstance(judgments, list):
            raise ValueError("Judgment file must contain a 'queries' list.")
        by_id = {entry.get("id"): entry for entry in judgments}
        if len(by_id) != len(judgments):
            raise ValueError("Judgment query IDs must be unique.")
        unknown = set(by_id) - seen_ids
        if unknown:
            raise ValueError(f"Judgments contain unknown query IDs: {sorted(unknown)}")
        for item in data:
            judgment = by_id.get(item["id"])
            if judgment is None:
                item["relevant_chunk_ids"] = None
                item["judgments_complete"] = False
                continue
            relevant_ids = judgment.get("relevant_chunk_ids")
            reviewed = judgment.get("candidate_pool_reviewed", False)
            if not isinstance(reviewed, bool):
                raise ValueError(
                    f"Query {item['id']} candidate_pool_reviewed must be a boolean."
                )
            if reviewed:
                if not isinstance(relevant_ids, list):
                    raise ValueError(
                        f"Reviewed query {item['id']} must have a relevant_chunk_ids list."
                    )
                candidate_ids = {
                    candidate["chunk_id"] for candidate in judgment.get("candidates", [])
                }
                unknown_chunks = set(relevant_ids) - candidate_ids
                if unknown_chunks:
                    raise ValueError(
                        f"Query {item['id']} labels chunks outside its candidate pool: "
                        f"{sorted(unknown_chunks)}"
                    )
                item["relevant_chunk_ids"] = relevant_ids
                item["judgments_complete"] = True
            else:
                item["relevant_chunk_ids"] = None
                item["judgments_complete"] = False
    return data


def evaluate(dataset_path, output_path, k=5, repeats=3, judgments_path=None):
    if k < 1:
        raise ValueError("Evaluation K must be at least 1.")
    if repeats < 1:
        raise ValueError("Evaluation repeats must be at least 1.")
    dataset = _load_dataset(Path(dataset_path), judgments_path)
    client = connect()
    try:
        latest_versions = get_latest_versions(client)
        collection = client.collections.get("PaperChunk")
        model = encoder()
        prepared_queries = [
            (
                item,
                model.encode(
                    item["question"],
                    normalize_embeddings=True,
                ).tolist(),
                build_filters(
                    latest_versions,
                    arxiv_id=item.get("arxiv_id"),
                    version=item.get("version"),
                ),
            )
            for item in dataset
        ]
        measured = {
            config: {item["id"]: {"latency_samples_ms": []} for item, _, _ in prepared_queries}
            for config in CONFIGURATIONS
        }

        for mode, alpha in CONFIGURATIONS:
            item, vector, filters = prepared_queries[0]
            retrieve_chunks(
                collection,
                item["question"],
                vector,
                filters,
                mode=mode,
                alpha=alpha if alpha is not None else settings.hybrid_alpha,
                limit=k,
            )

        order = random.Random(42)
        for _ in range(repeats):
            for item, vector, filters in prepared_queries:
                configurations = list(CONFIGURATIONS)
                order.shuffle(configurations)
                for mode, alpha in configurations:
                    started = time.perf_counter()
                    response = retrieve_chunks(
                        collection,
                        item["question"],
                        vector,
                        filters,
                        mode=mode,
                        alpha=alpha if alpha is not None else settings.hybrid_alpha,
                        limit=k,
                    )
                    measured[(mode, alpha)][item["id"]].update({
                        "response": response,
                        "latency_samples_ms": (
                            measured[(mode, alpha)][item["id"]]["latency_samples_ms"]
                            + [(time.perf_counter() - started) * 1000]
                        ),
                    })

        results = []
        for mode, alpha in CONFIGURATIONS:
            query_results = []
            for item, _, _ in prepared_queries:
                measurement = measured[(mode, alpha)][item["id"]]
                latency_samples_ms = measurement["latency_samples_ms"]
                retrieved = []
                for obj in measurement["response"].objects:
                    props = obj.properties
                    retrieved.append({
                        "chunk_id": str(obj.uuid),
                        "arxiv_id": props["arxiv_id"],
                        "version": props["version"],
                        "page": props["page"],
                        "section": props["section"],
                        "text": props["text"],
                    })
                query_results.append({
                    "id": item["id"],
                    "relevant_chunk_ids": item.get("relevant_chunk_ids"),
                    "judgments_complete": item.get("judgments_complete", False),
                    "retrieved_chunk_ids": [result["chunk_id"] for result in retrieved],
                    "results": retrieved,
                    "latency_samples_ms": latency_samples_ms,
                    "median_latency_ms": statistics.median(latency_samples_ms),
                })
            results.append({
                "mode": mode,
                "alpha": alpha,
                "metrics": _metrics(query_results, k),
                "latency_ms": {
                    "median": statistics.median(
                        item["median_latency_ms"] for item in query_results
                    ),
                    "mean": statistics.mean(
                        item["median_latency_ms"] for item in query_results
                    ),
                    "repeat_count": repeats,
                    "measurement_scope": "Weaviate retrieval call; excludes embedding generation",
                },
                "queries": query_results,
            })
    finally:
        client.close()

    output = {
        "k": k,
        "latency_repeats": repeats,
        "dataset": str(Path(dataset_path)),
        "judgments": str(Path(judgments_path)) if judgments_path else None,
        "judgment_scope": (
            "pooled top-K candidates returned by the evaluated configurations"
            if judgments_path
            else None
        ),
        "indexed_object_uuid": True,
        "metric_note": (
            "Recall@K is relative to the manually judged pooled candidate set, not "
            "all relevant chunks in the corpus. Precision@K is calculated only for "
            "queries whose entire pooled candidate set has been reviewed."
            if judgments_path
            else "Metrics are null until relevant chunk IDs are manually labeled."
        ),
        "results": results,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    _print_summary(output)
    print(f"\nSaved raw retrieval results to {output_path}")
    return output


def _print_summary(output):
    recall_name = "PoolRecall@K" if output.get("judgment_scope") else "Recall@K"
    print(f"Mode     Alpha  {recall_name:<11} Hit Rate@K  MRR       Precision@K  Median ms")
    for result in output["results"]:
        alpha = "N/A" if result["alpha"] is None else f"{result['alpha']:.2f}"
        metrics = result["metrics"]
        values = [
            "N/A" if metrics[key] is None else f"{metrics[key]:.3f}"
            for key in ("recall_at_k", "hit_rate_at_k", "mrr", "precision_at_k")
        ]
        latency = result["latency_ms"]["median"]
        print(f"{result['mode']:<8} {alpha:<6} " + "  ".join(values) + f"  {latency:.2f}")
    if all(result["metrics"]["relevant_queries"] == 0 for result in output["results"]):
        print("\nNo relevance labels were provided; these are retrieval rankings, not accuracy results.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="tests/data/retrieval_eval.json",
        help="JSON evaluation questions and optional manual relevance labels.",
    )
    parser.add_argument(
        "--output",
        default="data/retrieval_eval_results.json",
        help="Path for the raw retrieval results JSON.",
    )
    parser.add_argument(
        "--judgments",
        help="Manually reviewed JSON exported by scripts.prepare_retrieval_judgments.",
    )
    parser.add_argument("--k", type=int, default=5, help="Cutoff used for retrieval metrics.")
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Timed retrieval repeats per query and configuration (default: 3).",
    )
    args = parser.parse_args()
    evaluate(args.dataset, args.output, args.k, args.repeats, args.judgments)


if __name__ == "__main__":
    main()
