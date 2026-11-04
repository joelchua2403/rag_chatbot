
"""PDF extraction and retrieval chunking."""

import hashlib
import re

import pymupdf


HEADING = re.compile(
    r"^(?:\d+(?:\.\d+)*\s+)?"
    r"(?:abstract|introduction|background|related work|"
    r"methodology|methods|experiments|results|discussion|"
    r"conclusion|limitations|references)\b",
    re.IGNORECASE,
)

ARXIV_WATERMARK = re.compile(
    r"arXiv:\d{4}\.\d{4,5}v\d+",
    re.IGNORECASE,
)


def clean_block(text):
    """Clean layout noise without removing scientific content."""

    text = ARXIV_WATERMARK.sub("", text)

    # Remove standalone arXiv category labels.
    text = re.sub(
        r"(?m)^\s*\[(?:quant-ph|cs\.[A-Za-z]+)\]\s*$",
        "",
        text,
    )

    # Join hyphenated line breaks.
    text = re.sub(
        r"(\w)-\s*\n\s*(\w)",
        r"\1\2",
        text,
    )

    # Join ordinary wrapped lines within a block.
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def extract_pages(path):
    """
    Extract PDF text using block coordinates.

    Handles simple two-column academic layouts while
    preserving page metadata.
    """

    pages = []

    with pymupdf.open(path) as doc:
        for page_index, page in enumerate(doc):

            width = page.rect.width
            height = page.rect.height

            raw_blocks = page.get_text("blocks")

            blocks = []

            for block in raw_blocks:
                x0, y0, x1, y1, text, *_ = block

                if not isinstance(text, str):
                    continue

                # Exclude likely top/bottom page furniture.
                # These thresholds are deliberately conservative.
                if y1 < height * 0.035:
                    continue

                if y0 > height * 0.97:
                    continue

                cleaned = clean_block(text)

                if not cleaned:
                    continue

                blocks.append({
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "text": cleaned,
                })

            if not blocks:
                continue

            # Separate blocks by horizontal position.
            # A simple heuristic for two-column layouts.
            left = []
            right = []
            spanning = []

            mid = width / 2

            for block in blocks:
                if block["x0"] < mid and block["x1"] > mid:
                    spanning.append(block)
                elif block["x1"] <= mid:
                    left.append(block)
                else:
                    right.append(block)

            left.sort(key=lambda b: (b["y0"], b["x0"]))
            right.sort(key=lambda b: (b["y0"], b["x0"]))
            spanning.sort(key=lambda b: (b["y0"], b["x0"]))

            # Full-width blocks often contain titles or
            # section headings, so place early spanning
            # blocks before column text.
            first_column_y = min(
                [b["y0"] for b in left + right],
                default=height,
            )

            top_spanning = [
                b for b in spanning
                if b["y0"] <= first_column_y
            ]

            other_spanning = [
                b for b in spanning
                if b["y0"] > first_column_y
            ]

            # Order columns left-to-right.
            ordered = (
                top_spanning
                + left
                + right
                + other_spanning
            )

            page_text = "\n\n".join(
                block["text"] for block in ordered
            )

            if page_text.strip():
                pages.append({
                    "page": page_index + 1,
                    "text": page_text,
                })

    if not pages:
        raise ValueError(
            f"No extractable text in {path}"
        )

    return pages


def chunk_pages(pages, size=900, overlap=120):
    """Create overlapping chunks for semantic retrieval."""

    if size < 100 or not 0 <= overlap < size:
        raise ValueError("Invalid chunk size/overlap")

    chunks = []
    section = "Document"

    for page in pages:

        text = page["text"]

        for line in text.splitlines():
            candidate = line.strip()

            if (
                len(candidate) < 110
                and HEADING.match(candidate)
            ):
                section = candidate

        start = 0

        while start < len(text):

            end = min(
                len(text),
                start + size,
            )

            if end < len(text):
                boundary = text.rfind(
                    " ",
                    start + int(size * 0.7),
                    end,
                )

                if boundary > start:
                    end = boundary

            part = text[start:end].strip()

            if part:
                chunks.append({
                    "page": page["page"],
                    "section": section,
                    "text": part,
                    "content_hash": hashlib.sha256(
                        part.encode()
                    ).hexdigest(),
                })

            if end >= len(text):
                break

            start = max(
                start + 1,
                end - overlap,
            )

    return chunks
