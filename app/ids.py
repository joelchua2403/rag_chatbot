from uuid import uuid5, NAMESPACE_URL

def version_uuid(arxiv_id: str, version: int) -> str:
    return str(uuid5(NAMESPACE_URL, f"paper:{arxiv_id}:v{version}"))

def chunk_uuid(arxiv_id: str, version: int, index: int) -> str:
    return str(uuid5(NAMESPACE_URL, f"chunk:{arxiv_id}:v{version}:{index}"))
