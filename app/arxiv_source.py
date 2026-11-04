"""arXiv Atom API + export.arxiv.org version-specific PDF downloads."""
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
import httpx

PAPERS = ["2605.18821", "2606.05387", "2607.00365"]
ATOM = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
ID_PATTERN = re.compile(r"^\d{4}\.\d{4,5}$")


def validate_id(arxiv_id: str):
    if not ID_PATTERN.fullmatch(arxiv_id):
        raise ValueError(f"Invalid modern arXiv ID: {arxiv_id}")


def get_versions(arxiv_id: str) -> list[dict]:
    validate_id(arxiv_id)
    # arXiv export API version history is represented by the arxiv:version elements.
    url = "https://export.arxiv.org/api/query"
    with httpx.Client(timeout=60, follow_redirects=True, headers={"User-Agent": "AssessmentRAG/1.0 (academic demo)"}) as client:
        r = client.get(url, params={"id_list": arxiv_id})
        r.raise_for_status()
    root = ET.fromstring(r.content)
    entry = root.find("a:entry", ATOM)
    if entry is None or entry.find("a:title", ATOM) is None:
        raise RuntimeError(f"No arXiv record found for {arxiv_id}")
    title = " ".join(entry.findtext("a:title", default="", namespaces=ATOM).split())
    versions = []
    # arXiv API exposes version history through arxiv:version nodes when available.
    for node in entry.findall("arxiv:version", ATOM):
        number = node.attrib.get("version", "")
        if number.startswith("v") and number[1:].isdigit():
            versions.append({"version": int(number[1:]), "date": node.findtext("a:date", default="", namespaces=ATOM)})
    # Fallback: newest version can be inferred from the versioned entry id.
    if not versions:
        entry_id = entry.findtext("a:id", default="", namespaces=ATOM)
        match = re.search(r"v(\d+)$", entry_id)
        latest = int(match.group(1)) if match else 1
        versions = [{"version": n, "date": ""} for n in range(1, latest + 1)]
    return [{"arxiv_id": arxiv_id, "title": title, **item} for item in sorted(versions, key=lambda x: x["version"])]


def download_pdf(arxiv_id: str, version: int, destination: Path) -> Path:
    validate_id(arxiv_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 1000:
        return destination
    url = f"https://arxiv.org/pdf/{arxiv_id}v{version}"
    with httpx.Client(timeout=120, follow_redirects=True, headers={"User-Agent": "AssessmentRAG/1.0 (academic demo)"}) as client:
        r = client.get(url)
        r.raise_for_status()
    if not r.content.startswith(b"%PDF"):
        raise ValueError(f"Non-PDF response from {url}")
    destination.write_bytes(r.content)
    time.sleep(1)
    return destination
