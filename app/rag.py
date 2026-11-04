
import re

from sentence_transformers import SentenceTransformer
from weaviate.classes.query import Filter

from app.config import settings
from app.llm import generate
from app.store import connect


_encoder = None

ARXIV_PATTERN = re.compile(
    r"\b(\d{4}\.\d{4,5})(?:v(\d+))?\b",
    re.IGNORECASE,
)

CITATION_PATTERN = re.compile(r"\[S(\d+)\]")


def encoder():
    global _encoder

    if _encoder is None:
        _encoder = SentenceTransformer(
            settings.embedding_model
        )

    return _encoder


def parse_paper_reference(question: str):
    """Extract arXiv ID and optional version from a question."""
    match = ARXIV_PATTERN.search(question)

    if not match:
        return None, None

    arxiv_id = match.group(1)
    version = int(match.group(2)) if match.group(2) else None

    return arxiv_id, version


def get_latest_versions(client):
    """
    Return latest version per paper.

    Example:
    {
        "2605.18821": 1,
        "2606.05387": 2,
        "2607.00365": 1,
    }
    """
    collection = client.collections.get("PaperVersion")

    response = collection.query.fetch_objects(
        limit=1000
    )

    latest = {}

    for obj in response.objects:
        props = obj.properties

        arxiv_id = props["arxiv_id"]
        version = int(props["version"])

        latest[arxiv_id] = max(
            latest.get(arxiv_id, 0),
            version,
        )

    return latest




def detect_requested_version(
    question: str,
    arxiv_id: str | None = None,
) -> int | None:
    """
    Detect explicit arXiv version requests.

    Examples:
    - "version 1 of arXiv 2606.05387"
    - "arXiv 2606.05387v1"
    - "paper 2606.05387 version 2"
    - "according to v2 of this paper"
    """

    # arXiv ID with attached version.
    match = re.search(
        r"\b\d{4}\.\d{4,5}v(\d+)\b",
        question,
        re.IGNORECASE,
    )

    if match:
        return int(match.group(1))

    # Explicit version wording.
    match = re.search(
        r"\bversion\s+(\d+)\b",
        question,
        re.IGNORECASE,
    )

    if match:
        return int(match.group(1))

    # Short version notation.
    match = re.search(
        r"\bv(\d+)\b",
        question,
        re.IGNORECASE,
    )

    if match:
        return int(match.group(1))

    return None



def build_filters(
    latest_versions,
    arxiv_id=None,
    version=None,
):
    """Build Weaviate filters for valid paper versions."""

    if arxiv_id is not None:
        if arxiv_id not in latest_versions:
            raise ValueError(
                f"Paper {arxiv_id} not found in the corpus."
            )

        selected_version = (
            version
            if version is not None
            else latest_versions[arxiv_id]
        )

        return (
            Filter.by_property("arxiv_id").equal(arxiv_id)
            & Filter.by_property("version").equal(
                selected_version
            )
        )

    if version is not None:
        raise ValueError(
            "Version filtering requires an arxiv_id."
        )

    # Restrict corpus-wide retrieval to the latest
    # version of each paper.
    version_keys = [
        f"{paper_id}v{latest_version}"
        for paper_id, latest_version in latest_versions.items()
    ]

    if not version_keys:
        return None

    return Filter.by_property("version_key").contains_any(
        version_keys
    )



def validate_citations(response, sources):
    """Extract individual and grouped citation references."""

    citation_groups = re.findall(
        r"\[(S\d+(?:\s*,\s*S\d+)*)\]",
        response
    )

    cited_labels = set()

    for group in citation_groups:
        cited_labels.update(
            label.strip()
            for label in group.split(",")
        )

    available_labels = {
        source["label"]
        for source in sources
    }

    invalid = cited_labels - available_labels

    if invalid:
        raise ValueError(
            f"Invalid citations: {sorted(invalid)}"
        )

    return [
        source
        for source in sources
        if source["label"] in cited_labels
    ]



def answer(
    question: str,
    arxiv_id=None,
    version=None,
):
    """
    Answer questions using Weaviate retrieval.

    Default behavior:
    - Detect paper IDs automatically.
    - Retrieve latest versions.
    - Filter unrelated papers.
    - Return only cited sources.
    -Use [S1] [S2] instead of grouped citations.
    """

    detected_id, detected_version = parse_paper_reference(
        question
    )

    if arxiv_id is None:
        arxiv_id = detected_id

    if version is None:
        version = detect_requested_version(
        question,
        arxiv_id,
    )

    if version is not None and arxiv_id is None:
        raise ValueError(
            "Version filtering requires an arxiv_id."
        )

    vector = encoder().encode(
        question,
        normalize_embeddings=True,
    ).tolist()

    client = connect()

    try:
        latest_versions = get_latest_versions(client)

        filters = build_filters(
            latest_versions,
            arxiv_id=arxiv_id,
            version=version,
        )

        collection = client.collections.get("PaperChunk")

        result = collection.query.near_vector(
            near_vector=vector,
            limit=settings.retrieval_top_k,
            filters=filters,
        )

        sources = []

        for idx, obj in enumerate(
            result.objects,
            start=1,
        ):
            p = obj.properties

            sources.append({
                "label": f"S{idx}",
                "arxiv_id": p["arxiv_id"],
                "version": p["version"],
                "page": p["page"],
                "section": p["section"],
                "text": p["text"],
            })

    finally:
        client.close()

    if not sources:
        return {
            "answer": (
                "I could not find relevant indexed material."
            ),
            "sources": [],
        }

    context = "\n\n".join(
        (
            f"[{s['label']}] "
            f"arXiv:{s['arxiv_id']}v{s['version']} "
            f"page {s['page']}\n"
            f"{s['text']}"
        )
        for s in sources
    )

    system_prompt = """
You are an academic research assistant.

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

    response = generate(
        system_prompt,
        f"Question: {question}\n\nExcerpts:\n{context}",
    )

    used_sources = validate_citations(
        response,
        sources,
    )

    return {
        "answer": response,
        "sources": [
            {
                k: v
                for k, v in source.items()
                if k != "text"
            }
            for source in used_sources
        ],
    }
