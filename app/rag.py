"""Version-aware retrieval-augmented question answering for indexed arXiv papers."""

import re

from sentence_transformers import SentenceTransformer
from weaviate.classes.query import Filter

from app.comparison import compare
from app.config import settings
from app.llm import generate
from app.store import connect

_encoder = None

ARXIV_PATTERN = re.compile(r"\b(\d{4}\.\d{4,5})(?:v(\d+))?\b", re.IGNORECASE)
CITATION_PATTERN = re.compile(r"\[(S\d+(?:\s*,\s*S\d+)*)\]")
COMPARISON_INTENT_PATTERN = re.compile(
    r"\b(compare|comparison|differences?|changed|changes|revisions?|"
    r"what(?:'s| is) new)\b", re.IGNORECASE
)
VERSION_CONTEXT_PATTERN = re.compile(r"\bversions?\b|\bv\d+\b|\d{4}\.\d{4,5}v\d+", re.IGNORECASE)


def encoder():
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer(settings.embedding_model)
    return _encoder


def parse_paper_reference(question: str):
    match = ARXIV_PATTERN.search(question)
    if not match:
        return None, None
    return match.group(1), int(match.group(2)) if match.group(2) else None


def detect_requested_version(question: str, arxiv_id: str | None = None) -> int | None:
    """Select a single version for ordinary (non-comparison) retrieval."""
    match = re.search(r"\b\d{4}\.\d{4,5}v(\d+)\b", question, re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"\bversion\s+(\d+)\b", question, re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"\bv(\d+)\b", question, re.IGNORECASE)
    return int(match.group(1)) if match else None


def detect_comparison_intent(question: str) -> bool:
    return bool(COMPARISON_INTENT_PATTERN.search(question)) and bool(
        VERSION_CONTEXT_PATTERN.search(question)
    )


def detect_comparison_versions(question: str) -> tuple[int, int] | None:
    """Find explicitly named version pairs; None uses the latest two indexed versions."""
    patterns = (
        r"\bversions?\s+(\d+)\s+(?:and|to|vs\.?|versus)\s+(?:versions?\s+)?(\d+)\b",
        r"\bv(\d+)\s+(?:and|to|vs\.?|versus)\s+v(\d+)\b",
        r"\bversion\s+(\d+).*?\bversion\s+(\d+)\b",
        r"\bfrom\s+v?(\d+)\s+to\s+v?(\d+)\b",
        r"\b\d{4}\.\d{4,5}v(\d+)\s+(?:and|to|vs\.?|versus)\s+v(\d+)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, question, re.IGNORECASE | re.DOTALL)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def answer_version_comparison(
    arxiv_id: str,
    old_version: int | None = None,
    new_version: int | None = None,
) -> dict:
    """Return comparison evidence without LLM re-interpretation."""
    report = compare(
        arxiv_id=arxiv_id,
        old_version=old_version,
        new_version=new_version,
    )
    lines = [
        f"Version comparison: arXiv:{report.arxiv_id}v{report.old_version} "
        f"vs v{report.new_version}",
        "",
        report.summary,
    ]
    if report.changes:
        lines.extend(["", "Detected changes:"])
        for index, change in enumerate(report.changes, start=1):
            lines.append(
                f"{index}. {change.section} ({change.category}, "
                f"{change.change_type}): {change.description}"
            )
    else:
        lines.extend(["", "No changed text blocks were detected."])
    # GapReport does not include page-level citation data.
    return {"answer": "\n".join(lines), "sources": []}


def get_latest_versions(client):
    """Return latest indexed version per paper (sufficient for this small corpus)."""
    collection = client.collections.get("PaperVersion")
    response = collection.query.fetch_objects(limit=1000)
    latest = {}
    for obj in response.objects:
        props = obj.properties
        paper_id = props["arxiv_id"]
        latest[paper_id] = max(latest.get(paper_id, 0), int(props["version"]))
    return latest


def build_filters(latest_versions, arxiv_id=None, version=None):
    if arxiv_id is not None:
        if arxiv_id not in latest_versions:
            raise ValueError(f"Paper {arxiv_id} not found in the corpus.")
        selected_version = version if version is not None else latest_versions[arxiv_id]
        if selected_version > latest_versions[arxiv_id] or selected_version < 1:
            raise ValueError(f"Version {selected_version} is not indexed for {arxiv_id}.")
        return (
            Filter.by_property("arxiv_id").equal(arxiv_id)
            & Filter.by_property("version").equal(selected_version)
        )
    if version is not None:
        raise ValueError("Version filtering requires an arxiv_id.")
    version_keys = [f"{paper_id}v{latest_version}" for paper_id, latest_version in latest_versions.items()]
    return Filter.by_property("version_key").contains_any(version_keys) if version_keys else None


def retrieve_chunks(
    collection,
    question: str,
    vector: list[float],
    filters,
    *,
    mode: str | None = None,
    alpha: float | None = None,
    limit: int | None = None,
):
    """Run dense or Weaviate-native hybrid retrieval with the supplied version filter."""
    mode = settings.retrieval_mode if mode is None else mode
    limit = settings.retrieval_top_k if limit is None else limit

    if mode == "dense":
        return collection.query.near_vector(
            near_vector=vector,
            limit=limit,
            filters=filters,
        )
    if mode == "hybrid":
        alpha = settings.hybrid_alpha if alpha is None else alpha
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Hybrid alpha must be between 0.0 and 1.0.")
        return collection.query.hybrid(
            query=question,
            vector=vector,
            alpha=alpha,
            limit=limit,
            filters=filters,
        )
    raise ValueError(f"Unsupported retrieval mode: {mode}")


def validate_citations(response, sources):
    cited_labels = set()
    for group in CITATION_PATTERN.findall(response):
        cited_labels.update(label.strip() for label in group.split(","))
    available = {source["label"] for source in sources}
    invalid = cited_labels - available
    if invalid:
        raise ValueError(f"Invalid citations: {sorted(invalid)}")
    return [source for source in sources if source["label"] in cited_labels]


def answer(question: str, arxiv_id=None, version=None):
    detected_id, detected_version = parse_paper_reference(question)
    if arxiv_id is None:
        arxiv_id = detected_id

    # Route comparisons BEFORE embedding and vector retrieval.
    if detect_comparison_intent(question):
        if arxiv_id is None:
            raise ValueError("Please specify an arXiv ID for version comparison.")
        versions = detect_comparison_versions(question)
        if versions is None:
            return answer_version_comparison(arxiv_id)
        old_version, new_version = versions
        if old_version >= new_version:
            raise ValueError("The older version must be less than the newer version.")
        return answer_version_comparison(
            arxiv_id=arxiv_id, old_version=old_version, new_version=new_version
        )

    if version is None:
        version = detected_version if detected_version is not None else detect_requested_version(question, arxiv_id)
    if version is not None and arxiv_id is None:
        raise ValueError("Version filtering requires an arxiv_id.")

    vector = encoder().encode(question, normalize_embeddings=True).tolist()
    client = connect()
    try:
        latest_versions = get_latest_versions(client)
        filters = build_filters(latest_versions, arxiv_id=arxiv_id, version=version)
        collection = client.collections.get("PaperChunk")
        result = retrieve_chunks(collection, question, vector, filters)
        sources = []
        for idx, obj in enumerate(result.objects, start=1):
            props = obj.properties
            sources.append({
                "label": f"S{idx}",
                "arxiv_id": props["arxiv_id"],
                "version": props["version"],
                "page": props["page"],
                "section": props["section"],
                "text": props["text"],
            })
    finally:
        client.close()

    if not sources:
        return {"answer": "I could not find relevant indexed material.", "sources": []}

    context = "\n\n".join(
        f"[{s['label']}] arXiv:{s['arxiv_id']}v{s['version']} "
        f"page {s['page']}\n{s['text']}"
        for s in sources
    )
    system_prompt = """You are an academic research assistant.
Answer questions using ONLY the supplied excerpts.
Rules:
1. Do not use external knowledge.
2. Cite each factual claim with [S1], [S2], etc.
3. Cite sources immediately after the claim.
4. Never invent citations.
5. If evidence is insufficient, say so.
6. Do not invent experiments, numbers or conclusions.
7. Do not combine findings from different versions.
8. Prefer concise and accurate explanations.
"""
    response = generate(system_prompt, f"Question: {question}\n\nExcerpts:\n{context}")
    used_sources = validate_citations(response, sources)
    return {
        "answer": response,
        "sources": [
            {key: value for key, value in source.items() if key != "text"}
            for source in used_sources
        ],
    }
