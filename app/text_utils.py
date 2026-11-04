
import re
import unicodedata


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)

    # Remove arXiv version watermarks.
    text = re.sub(
        r"arXiv:\d{4}\.\d{4,5}v\d+",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # Remove isolated arXiv category labels.
    text = re.sub(
        r"\[(?:quant-ph|cs\.[A-Za-z]+)\]",
        " ",
        text,
    )

    # Join hyphenated line breaks.
    text = re.sub(
        r"(\w)-\s*\n\s*(\w)",
        r"\1\2",
        text,
    )

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()
