from pathlib import Path
from sentence_transformers import SentenceTransformer
from app.config import settings
from app.arxiv_source import PAPERS, get_versions, download_pdf
from app.parsing import extract_pages, chunk_pages
from app.store import connect, setup


def ingest(ids=None):
    ids = ids or PAPERS
    encoder = SentenceTransformer(settings.embedding_model)
    client = connect()
    try:
        setup(client)
        from app.store import upsert_document
        for aid in ids:
            for meta in get_versions(aid):
                version = meta["version"]
                pdf = download_pdf(aid, version, Path(settings.data_dir) / f"{aid}v{version}.pdf")
                pages = extract_pages(pdf)
                chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
                embeddings = encoder.encode([c["text"] for c in chunks], normalize_embeddings=True).tolist()
                result = upsert_document(client, meta, pages, chunks, embeddings)
                print(f"{aid}v{version}: {result}")
    finally:
        client.close()
