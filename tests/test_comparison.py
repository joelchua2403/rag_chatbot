
import pytest

from app.comparison import (
    compute_diffs,
    classify,
    remove_moved_text_artifacts,
    remove_embedded_text_artifacts,
)


def test_numerical_change():
    changes = compute_diffs(
        "The result is 95%.",
        "The result is 97%.",
    )

    assert len(changes) == 1
    assert changes[0]["change_type"] == "replace"

    result = classify(changes[0])
    assert result["category"] == "substantive"


def test_formatting_only():
    changes = compute_diffs(
        "The model achieves\nhigh accuracy.",
        "The model achieves high accuracy.",
    )

    assert changes == []


def test_added_experiment():
    changes = compute_diffs(
        "We evaluate the model.",
        "We evaluate the model.\n\n"
        "We added an experiment using 100 samples.",
    )

    assert len(changes) == 1
    assert changes[0]["change_type"] == "insert"


def test_removed_claim():
    changes = compute_diffs(
        "The model is accurate.\n\n"
        "It outperforms existing methods.",
        "The model is accurate.",
    )

    assert len(changes) == 1
    assert changes[0]["change_type"] == "delete"


def test_publication_date_change():
    result = classify({
        "change_type": "replace",
        "before": "3 Jun 2026",
        "after": "28 Jul 2026",
    })

    assert result["category"] == "cosmetic"
    assert "[Metadata]" in result["description"]
    assert "3 Jun 2026" in result["description"]
    assert "28 Jul 2026" in result["description"]


def test_affiliation_change():
    result = classify({
        "change_type": "delete",
        "before": (
            "King Abdullah University of Science "
            "and Technology (KAUST), Saudi Arabia"
        ),
        "after": "",
    })

    assert result["category"] == "cosmetic"
    assert "[Metadata]" in result["description"]


def test_moved_text_not_scientific_change():
    sentence = (
        "The proposed quantum algorithm achieves "
        "a measurable improvement over the baseline."
    )

    changes = [
        {
            "change_type": "delete",
            "before": sentence,
            "after": "",
        },
        {
            "change_type": "insert",
            "before": "",
            "after": sentence,
        },
    ]

    cleaned = remove_moved_text_artifacts(changes)

    assert cleaned == []


def test_nisq_pdf_alignment_artifact():
    nisq = (
        "coined the nisq era to describe precisely "
        "the regime in which this tension is most "
        "acute: current devices have 50–1000"
    )

    changes = [
        {
            "change_type": "replace",
            "before": (
                "Author’s Contact Information: "
                "University A, author@example.com"
            ),
            "after": (
                nisq
                + "\n"
                + "Author’s Contact Information: "
                "University B, author@example.com"
            ),
        },
        {
            "change_type": "delete",
            "before": nisq,
            "after": "",
        },
    ]

    cleaned = remove_embedded_text_artifacts(changes)

    assert len(cleaned) == 1
    assert cleaned[0]["change_type"] == "replace"
    assert nisq not in cleaned[0]["after"]
    assert "University B" in cleaned[0]["after"]
