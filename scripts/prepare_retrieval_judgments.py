"""Export an editable pooled candidate set for manual relevance judgments."""

import argparse
import json
from pathlib import Path


def build_template(results_path, dataset_path):
    results = json.loads(Path(results_path).read_text(encoding="utf-8"))
    questions = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    questions_by_id = {item["id"]: item for item in questions}
    pooled = {}

    for configuration in results["results"]:
        for query_result in configuration["queries"]:
            query_id = query_result["id"]
            if query_id not in questions_by_id:
                raise ValueError(f"Evaluation result has unknown query ID: {query_id}")
            query_pool = pooled.setdefault(query_id, {})
            for rank, item in enumerate(query_result["results"], start=1):
                chunk_id = item["chunk_id"]
                candidate = query_pool.setdefault(
                    chunk_id,
                    {
                        "chunk_id": chunk_id,
                        "arxiv_id": item["arxiv_id"],
                        "version": item["version"],
                        "page": item["page"],
                        "section": item["section"],
                        "text": item["text"],
                        "retrieved_by": [],
                    },
                )
                candidate["retrieved_by"].append({
                    "mode": configuration["mode"],
                    "alpha": configuration["alpha"],
                    "rank": rank,
                })

    query_templates = []
    for item in questions:
        candidates = sorted(
            pooled.get(item["id"], {}).values(),
            key=lambda candidate: candidate["chunk_id"],
        )
        query_templates.append({
            "id": item["id"],
            "question": item["question"],
            "category": item["category"],
            "source_basis": item["source_basis"],
            "candidates": candidates,
            "relevant_chunk_ids": [],
            "candidate_pool_reviewed": False,
        })

    return {
        "instructions": (
            "For each query, inspect every candidate passage. Add the UUID of every "
            "candidate that directly supports answering the question to "
            "relevant_chunk_ids. After reviewing every candidate, set "
            "candidate_pool_reviewed to true. This is pooled evaluation: recall is "
            "relative to this candidate union, not the entire corpus."
        ),
        "pooling": "Union of top-K results from dense and each tested hybrid alpha.",
        "queries": query_templates,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        default="data/retrieval_eval_results.json",
        help="Previously generated retrieval results JSON.",
    )
    parser.add_argument(
        "--dataset",
        default="tests/data/retrieval_eval.json",
        help="Evaluation query dataset JSON.",
    )
    parser.add_argument(
        "--output",
        default="data/retrieval_judgments.json",
        help="Output path for the editable judgment template.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError(
            f"{output_path} already exists; choose a different --output path "
            "to avoid overwriting manual judgments."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(build_template(args.results, args.dataset), indent=2),
        encoding="utf-8",
    )
    print(f"Created judgment template: {output_path}")
    print("Review every candidate, enter relevant chunk UUIDs, then set candidate_pool_reviewed=true.")


if __name__ == "__main__":
    main()
