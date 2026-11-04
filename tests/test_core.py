from app.ids import version_uuid, chunk_uuid
from app.parsing import chunk_pages
from app.comparison import compute_diffs

def test_ids_are_deterministic_and_version_specific():
    assert version_uuid("2605.18821", 1) == version_uuid("2605.18821", 1)
    assert version_uuid("2605.18821", 1) != version_uuid("2605.18821", 2)
    assert chunk_uuid("2605.18821", 1, 0) != chunk_uuid("2605.18821", 1, 1)

def test_chunking():
    chunks = chunk_pages([{"page": 1, "text": "word " * 300}], size=200, overlap=30)
    assert len(chunks) > 1
    assert all(c["page"] == 1 for c in chunks)

def test_diff_detects_numeric_changes():
    changes = compute_diffs("Accuracy improved to 81 percent on the benchmark.\n\nThe method was tested.",
                            "Accuracy improved to 91 percent on the benchmark.\n\nThe method was tested.")
    assert any("81" in c["before"] and "91" in c["after"] for c in changes)
