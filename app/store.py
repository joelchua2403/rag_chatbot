import hashlib
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import Filter
from app.config import settings
from app.ids import version_uuid, chunk_uuid


def connect():
    return weaviate.connect_to_local(host=settings.weaviate_host,
                                    port=settings.weaviate_http_port,
                                    grpc_port=settings.weaviate_grpc_port)


def setup(client):
    if not client.collections.exists("PaperVersion"):
        client.collections.create(name="PaperVersion", vector_config=Configure.Vectors.self_provided(),
            properties=[Property(name=k, data_type=t) for k,t in [
                ("arxiv_id",DataType.TEXT),("version",DataType.INT),("version_key",DataType.TEXT),
                ("title",DataType.TEXT),("date",DataType.TEXT),("content_hash",DataType.TEXT),
                ("chunk_count",DataType.INT),("full_text",DataType.TEXT)]])
    if not client.collections.exists("PaperChunk"):
        client.collections.create(name="PaperChunk", vector_config=Configure.Vectors.self_provided(),
            properties=[Property(name=k, data_type=t) for k,t in [
                ("arxiv_id",DataType.TEXT),("version",DataType.INT),("version_key",DataType.TEXT),
                ("title",DataType.TEXT),("page",DataType.INT),("section",DataType.TEXT),
                ("chunk_index",DataType.INT)]] + [
                Property(name="text", data_type=DataType.TEXT, index_searchable=True),
                Property(name="content_hash", data_type=DataType.TEXT),
            ])


def version_filter(arxiv_id, version=None):
    f = Filter.by_property("arxiv_id").equal(arxiv_id)
    if version is not None:
        f = f & Filter.by_property("version").equal(version)
    return f



def upsert_document(client, metadata, pages, chunks, embeddings):
    aid = metadata["arxiv_id"]
    version = metadata["version"]
    key = f"{aid}v{version}"

    if len(chunks) != len(embeddings):
        raise ValueError("Chunk and embedding counts do not match")

    full_text = "\n\n".join(
        f"[Page {p['page']}]\n{p['text']}"
        for p in pages
    )

    digest = hashlib.sha256(
        (
            full_text
            + settings.embedding_model
            + str(settings.chunk_size)
            + str(settings.chunk_overlap)
        ).encode()
    ).hexdigest()

    versions = client.collections.get("PaperVersion")
    chunks_collection = client.collections.get("PaperChunk")

    doc_id = version_uuid(aid, version)

    existing = versions.query.fetch_object_by_id(doc_id)

    if existing and existing.properties.get("content_hash") == digest:
        return {
            "status": "skipped",
            "chunks": len(chunks),
        }

    # Remove old chunks belonging to this version.
    # This is more efficient than scanning all objects.
    chunks_collection.data.delete_many(
        where=version_filter(aid, version)
    )

    # Insert updated chunks and embeddings.
    for idx, (chunk, vector) in enumerate(
        zip(chunks, embeddings, strict=True)
    ):
        chunks_collection.data.insert(
            uuid=chunk_uuid(aid, version, idx),
            vector=vector,
            properties={
                "arxiv_id": aid,
                "version": version,
                "version_key": key,
                "title": metadata["title"],
                "page": chunk["page"],
                "section": chunk["section"],
                "chunk_index": idx,
                "text": chunk["text"],
                "content_hash": chunk["content_hash"],
            },
        )

    props = {
        "arxiv_id": aid,
        "version": version,
        "version_key": key,
        "title": metadata["title"],
        "date": metadata.get("date", ""),
        "content_hash": digest,
        "chunk_count": len(chunks),
        "full_text": full_text,
    }

    if existing:
        versions.data.replace(
            uuid=doc_id,
            properties=props,
        )
    else:
        versions.data.insert(
            uuid=doc_id,
            properties=props,
            vector=[1.0],
        )

    return {
        "status": "ingested",
        "chunks": len(chunks),
    }


def available_versions(client, aid):
    coll = client.collections.get("PaperVersion")
    return sorted([int(o.properties["version"]) for o in coll.iterator(return_properties=["arxiv_id","version"])
                   if o.properties["arxiv_id"] == aid])


def get_document(client, aid, version):
    obj = client.collections.get("PaperVersion").query.fetch_object_by_id(version_uuid(aid, version))
    return obj.properties if obj else None
