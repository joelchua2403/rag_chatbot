
"""
Evidence-grounded arXiv paper version comparison.

Pipeline:
1. Retrieve indexed paper versions from Weaviate.
2. Normalize PDF-extracted text.
3. Align paragraphs and detect changes.
4. Remove identical moved-text artifacts.
5. Remove duplicated fragments embedded in replacements.
6. Classify metadata and cosmetic changes deterministically.
7. Use an LLM for ambiguous scientific changes.
8. Generate a structured comparison report.

Limitations:
- PDF extraction can alter reading order.
- Moved text is treated as unchanged scientific content.
- Heuristic metadata detection is not universally reliable.
- LLM classifications require review for high-stakes use.
"""

import difflib
import json
import re
from collections import Counter

from app.llm import generate
from app.models import GapReport, Change
from app.store import connect, available_versions, get_document
from app.text_utils import normalize_text


# ==================================================
# 1. TEXT NORMALIZATION
# ==================================================

PAGE_MARKER = re.compile(r"\[Page \d+\]")

DATE_PATTERN = re.compile(
    r"\b\d{1,2}\s+"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"[a-z]*\s+\d{4}\b",
    re.IGNORECASE,
)

EMAIL_PATTERN = re.compile(
    r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"
)

NUMBER_PATTERN = re.compile(
    r"(?<![\w.])"
    r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?%?"
    r"(?![\w.])"
)

AFFILIATION_TERMS = (
    "università di pisa",
    "king abdullah university",
    "kaust",
    "saudi arabia",
    "author’s contact information",
    "author's contact information",
)

CONTACT_LABEL = "Author’s Contact Information"


def canonical(text: str) -> str:
    """Normalize text for exact content matching."""
    return " ".join(normalize_text(text).split())


def paragraphs(text: str) -> list[str]:
    """
    Split extracted document text into paragraphs.

    Preserve short paragraphs because they may contain
    publication dates, numerical results or headings.
    """

    text = PAGE_MARKER.sub("\n\n", text)

    blocks = re.split(r"\n\s*\n", text)

    result = []

    for block in blocks:
        normalized = normalize_text(block)

        if normalized:
            result.append(normalized)

    return result


# ==================================================
# 2. DIFFERENCE DETECTION
# ==================================================

def compute_diffs(
    old_text: str,
    new_text: str,
) -> list[dict]:
    """
    Compute deterministic paragraph-level differences.

    SequenceMatcher identifies unchanged paragraphs
    and groups insertions, deletions and replacements.
    """

    old_paragraphs = paragraphs(old_text)
    new_paragraphs = paragraphs(new_text)

    matcher = difflib.SequenceMatcher(
        a=old_paragraphs,
        b=new_paragraphs,
        autojunk=False,
    )

    changes = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():

        if tag == "equal":
            continue

        old_block = old_paragraphs[i1:i2]
        new_block = new_paragraphs[j1:j2]

        if tag == "replace":
            inner = difflib.SequenceMatcher(
                a=old_block,
                b=new_block,
                autojunk=False,
            )

            for inner_tag, x1, x2, y1, y2 in inner.get_opcodes():

                if inner_tag == "equal":
                    continue

                before = "\n".join(old_block[x1:x2])
                after = "\n".join(new_block[y1:y2])

                changes.append({
                    "change_type": inner_tag,
                    "before": before,
                    "after": after,
                })

        else:
            changes.append({
                "change_type": tag,
                "before": "\n".join(old_block),
                "after": "\n".join(new_block),
            })

    return changes


# ==================================================
# 3. REMOVE IDENTICAL MOVED-TEXT ARTIFACTS
# ==================================================

def remove_moved_text_artifacts(
    changes: list[dict],
) -> list[dict]:
    """
    Remove matching insert/delete pairs.

    Example:

    DELETE: "The model achieves 95% accuracy."
    INSERT: "The model achieves 95% accuracy."

    These represent unchanged content at a
    different location, not a scientific revision.
    """

    deleted = Counter(
        canonical(change["before"])
        for change in changes
        if change["change_type"] == "delete"
    )

    inserted = Counter(
        canonical(change["after"])
        for change in changes
        if change["change_type"] == "insert"
    )

    matched = deleted & inserted

    remaining_insertions = dict(matched)
    removed_deletions = Counter()

    cleaned = []

    for change in changes:

        change_type = change["change_type"]

        if change_type == "delete":
            key = canonical(change["before"])

            if removed_deletions[key] < matched[key]:
                removed_deletions[key] += 1
                continue

        if change_type == "insert":
            key = canonical(change["after"])

            if remaining_insertions.get(key, 0) > 0:
                remaining_insertions[key] -= 1
                continue

        cleaned.append(dict(change))

    return cleaned


# ==================================================
# 4. REMOVE EMBEDDED PDF ALIGNMENT ARTIFACTS
# ==================================================

def remove_embedded_text_artifacts(
    changes: list[dict],
) -> list[dict]:
    """
    Remove a duplicate insert/delete fragment when
    the same text appears inside the opposite side
    of another changed block.

    Also removes that duplicated fragment from the
    replacement block, preserving the remaining
    actual change.

    This is conservative: only exact normalized
    matches of at least 60 characters are handled.
    """

    result = [dict(change) for change in changes]

    removed_indices = set()

    for index, change in enumerate(result):

        change_type = change["change_type"]

        if change_type not in ("insert", "delete"):
            continue

        source_field = (
            "before"
            if change_type == "delete"
            else "after"
        )

        opposite_field = (
            "after"
            if change_type == "delete"
            else "before"
        )

        fragment = canonical(change[source_field])

        if len(fragment) < 60:
            continue

        for other_index, other in enumerate(result):

            if index == other_index:
                continue

            if other_index in removed_indices:
                continue

            if other["change_type"] != "replace":
                continue

            target = canonical(other[opposite_field])

            if fragment not in target:
                continue

            # Remove the duplicated text from the
            # replacement side where it was absorbed.
            cleaned_target = target.replace(
                fragment,
                "",
                1,
            ).strip()

            result[other_index][opposite_field] = (
                cleaned_target
            )

            removed_indices.add(index)
            break

    cleaned = []

    for index, change in enumerate(result):

        if index in removed_indices:
            continue

        before = change["before"].strip()
        after = change["after"].strip()

        if not before and not after:
            continue

        if canonical(before) == canonical(after):
            continue

        if not before:
            change_type = "insert"
        elif not after:
            change_type = "delete"
        else:
            change_type = "replace"

        cleaned.append({
            "change_type": change_type,
            "before": before,
            "after": after,
        })

    return cleaned


# ==================================================
# 5. METADATA DETECTION
# ==================================================

def detect_change_scope(change: dict) -> str:
    """
    Detect obvious publication metadata changes.

    Institution-name matching is heuristic and
    should not be used as the only criterion
    in a general-purpose scientific comparator.
    """

    before = change["before"]
    after = change["after"]

    combined = f"{before} {after}"
    lowered = combined.lower().strip()

    if not lowered:
        return "unknown"

    if "author’s contact information" in lowered:
        return "metadata"

    if "author's contact information" in lowered:
        return "metadata"

    if EMAIL_PATTERN.search(combined):
        return "metadata"

    # Standalone dates.
    if (
        DATE_PATTERN.fullmatch(before.strip())
        or DATE_PATTERN.fullmatch(after.strip())
    ):
        return "metadata"

    # Author affiliation text.
    if (
        len(combined) < 350
        and any(
            term in lowered
            for term in AFFILIATION_TERMS
        )
        and not re.search(
            r"\b(?:experiment|dataset|accuracy|result|method)\b",
            lowered,
        )
    ):
        return "metadata"

    return "scientific_or_cosmetic"


def metadata_description(change: dict) -> str:
    """
    Describe metadata changes deterministically,
    without asking an LLM to infer explanations.
    """

    before = change["before"].strip()
    after = change["after"].strip()

    combined = f"{before} {after}".lower()

    # Publication date replacement.
    if (
        DATE_PATTERN.fullmatch(before)
        and DATE_PATTERN.fullmatch(after)
    ):
        return (
            f"Publication date changed from "
            f"{before} to {after}."
        )

    # Date removed.
    if DATE_PATTERN.fullmatch(before):
        return (
            f"Publication metadata removed: {before}."
        )

    # Date added.
    if DATE_PATTERN.fullmatch(after):
        return (
            f"Publication metadata added: {after}."
        )

    # Contact details updated.
    if "contact information" in combined:

        old_emails = set(
            EMAIL_PATTERN.findall(before)
        )

        new_emails = set(
            EMAIL_PATTERN.findall(after)
        )

        removed_emails = sorted(
            old_emails - new_emails
        )

        added_emails = sorted(
            new_emails - old_emails
        )

        if not removed_emails and not added_emails:
            return (
                "Author contact or institutional "
                "affiliation details changed, while "
                "the listed email addresses remained "
                "unchanged."
            )

        details = []

        if removed_emails:
            details.append(
                "Removed email addresses: "
                + ", ".join(removed_emails)
            )

        if added_emails:
            details.append(
                "Added email addresses: "
                + ", ".join(added_emails)
            )

        return (
            "Author contact information changed. "
            + ". ".join(details)
            + "."
        )

    # Metadata added.
    if not before and after:
        return (
            "Author or institutional metadata added: "
            f"{after[:180]}"
        )

    # Metadata removed.
    if before and not after:
        return (
            "Author or institutional metadata removed: "
            f"{before[:180]}"
        )

    # Metadata replaced.
    if before and after:
        return (
            "Author or institutional metadata updated. "
            f"Before: {before[:120]}. "
            f"After: {after[:120]}."
        )

    return (
        "Metadata change detected; "
        "manual review required."
    )


# ==================================================
# 6. COSMETIC AND NUMERICAL DETECTION
# ==================================================

def extract_numbers(text: str) -> list[str]:
    """Extract numeric tokens from changed text."""
    return NUMBER_PATTERN.findall(text)


def is_cosmetic_change(
    before: str,
    after: str,
) -> bool:
    """
    Detect punctuation, whitespace and case changes.

    Does not intentionally remove digits or
    scientific symbols.
    """

    if not before or not after:
        return False

    def simplify(text):
        text = text.lower()
        text = re.sub(
            r"""[\s.,;:!?"'‘’“”]+""",
            "",
            text,
        )
        return text

    return simplify(before) == simplify(after)


# ==================================================
# 7. LLM CLASSIFICATION
# ==================================================

CLASSIFICATION_PROMPT = """
You are a scientific document comparison analyst.

Classify a change between two versions of
an academic paper.

Use ONLY the BEFORE and AFTER evidence.

Rules:

1. Never invent claims, numbers or experiments.
2. Never describe text as removed if it is
   present in the AFTER evidence.
3. Scientific claims, results, experiments,
   methods and conclusions are substantive.
4. Grammar, spelling, punctuation and
   whitespace changes are cosmetic.
5. Author information, affiliations and
   publication dates are metadata, not
   scientific findings.
6. Do not speculate about author intentions.
7. For insertions and deletions, describe
   only the evidence provided.
8. If uncertain, recommend manual review.

Return ONLY valid JSON:

{
  "category": "substantive or cosmetic",
  "description": "Short evidence-grounded explanation",
  "section": "Unspecified"
}
"""


def classify(change: dict) -> dict:
    """
    Classify a detected change.

    The existing Change schema supports two categories:
    - substantive
    - cosmetic

    Metadata is represented as cosmetic with a
    [Metadata] prefix to preserve API compatibility.
    """

    before = change["before"]
    after = change["after"]

    # ----------------------------------------------
    # Metadata
    # ----------------------------------------------

    scope = detect_change_scope(change)

    if scope == "metadata":
        section = (
            CONTACT_LABEL
            if "contact information" in
            f"{before} {after}".lower()
            else "Publication Metadata"
        )

        return {
            "category": "cosmetic",
            "description": (
                "[Metadata] "
                + metadata_description(change)
            ),
            "section": section,
        }

    # ----------------------------------------------
    # Cosmetic
    # ----------------------------------------------

    if is_cosmetic_change(before, after):
        return {
            "category": "cosmetic",
            "description": (
                "Only punctuation, spacing or "
                "letter case changed."
            ),
            "section": "Unspecified",
        }

    # ----------------------------------------------
    # Numerical differences
    # ----------------------------------------------

    if before and after:

        old_numbers = extract_numbers(before)
        new_numbers = extract_numbers(after)

        if old_numbers != new_numbers:
            return {
                "category": "substantive",
                "description": (
                    "Numerical values differ between "
                    "the two versions. "
                    f"Before: {old_numbers}; "
                    f"After: {new_numbers}. "
                    "Review whether these values "
                    "represent scientific findings."
                ),
                "section": "Unspecified",
            }

    # ----------------------------------------------
    # Ambiguous changes: use Ollama
    # ----------------------------------------------

    try:
        raw = generate(
            CLASSIFICATION_PROMPT,
            (
                f"BEFORE:\n{before[:5000]}\n\n"
                f"AFTER:\n{after[:5000]}"
            ),
            json_mode=True,
        )

        parsed = json.loads(raw)

        category = parsed.get("category")

        if category not in (
            "substantive",
            "cosmetic",
        ):
            raise ValueError(
                "Invalid classification"
            )

        description = str(
            parsed.get("description", "")
        ).strip()

        if not description:
            raise ValueError(
                "Missing description"
            )

        return {
            "category": category,
            "description": description,
            "section": str(
                parsed.get(
                    "section",
                    "Unspecified",
                )
            ),
        }

    except Exception:
        return {
            "category": "substantive",
            "description": (
                "Text changed, but automated "
                "classification was inconclusive. "
                "Manual review is required."
            ),
            "section": "Unspecified",
        }


# ==================================================
# 8. MAIN VERSION COMPARISON
# ==================================================

def compare(
    arxiv_id: str,
    old_version: int | None = None,
    new_version: int | None = None,
    max_changes: int = 30,
) -> GapReport:
    """
    Compare two indexed arXiv paper versions.

    Defaults:
    - new_version: latest indexed version
    - old_version: preceding indexed version
    """

    if max_changes < 1:
        raise ValueError(
            "max_changes must be positive"
        )

    client = connect()

    try:
        versions = available_versions(
            client,
            arxiv_id,
        )

        if new_version is None:

            if len(versions) < 2:
                raise ValueError(
                    f"{arxiv_id} has fewer than "
                    f"two indexed versions: {versions}"
                )

            new_version = versions[-1]

        if old_version is None:

            prior = [
                version
                for version in versions
                if version < new_version
            ]

            if not prior:
                raise ValueError(
                    "No prior indexed version"
                )

            old_version = prior[-1]

        if old_version >= new_version:
            raise ValueError(
                "old_version must be less "
                "than new_version"
            )

        old = get_document(
            client,
            arxiv_id,
            old_version,
        )

        new = get_document(
            client,
            arxiv_id,
            new_version,
        )

        if not old or not new:
            raise ValueError(
                "Requested versions are "
                "not both indexed"
            )

    finally:
        client.close()

    # ----------------------------------------------
    # Detect raw changes
    # ----------------------------------------------

    raw_diffs = compute_diffs(
        old["full_text"],
        new["full_text"],
    )

    # ----------------------------------------------
    # Remove exact moved text
    # ----------------------------------------------

    diffs = remove_moved_text_artifacts(
        raw_diffs
    )

    # ----------------------------------------------
    # Remove embedded duplicates
    # ----------------------------------------------

    diffs = remove_embedded_text_artifacts(
        diffs
    )

    # ----------------------------------------------
    # Classify changes
    # ----------------------------------------------

    classified_all = []

    for diff in diffs:

        label = classify(diff)

        classified_all.append(
            Change(
                **diff,
                **label,
            )
        )

    # ----------------------------------------------
    # Calculate summary counts
    # ----------------------------------------------

    substantive = sum(
        change.category == "substantive"
        for change in classified_all
    )

    metadata = sum(
        change.description.startswith(
            "[Metadata]"
        )
        for change in classified_all
    )

    cosmetic = (
        len(classified_all)
        - substantive
        - metadata
    )

    displayed = classified_all[:max_changes]

    # ----------------------------------------------
    # Build summary
    # ----------------------------------------------

    summary = (
        f"Compared {arxiv_id}v{old_version} "
        f"against v{new_version}. "
        f"Detected {len(diffs)} changed text blocks: "
        f"{substantive} classified as substantive, "
        f"{metadata} metadata changes, and "
        f"{cosmetic} other cosmetic changes. "
    )

    if substantive == 0:
        summary += (
            "No substantive scientific changes "
            "were identified by the current "
            "comparison pipeline. "
        )

    if len(diffs) > max_changes:
        summary += (
            f"Only the first {max_changes} changes "
            "are included in this response. "
        )

    summary += (
        "PDF extraction, paragraph alignment "
        "and automated classification may "
        "miss or mischaracterize changes; "
        "review the evidence."
    )

    return GapReport(
        arxiv_id=arxiv_id,
        old_version=old_version,
        new_version=new_version,
        changes=displayed,
        summary=summary,
    )
