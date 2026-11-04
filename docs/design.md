# Architecture and design notes (draft; adapt after actual evaluation)

## Design
The ingestion process discovers available versions of each supplied arXiv paper, downloads version-specific PDFs, extracts page text with PyMuPDF, creates overlapping page-local chunks, embeds them locally with SentenceTransformers, and stores self-provided vectors in Weaviate. Two collections are used: PaperVersion (document-level metadata, full extracted text, content hash) and PaperChunk (filterable arXiv ID/version, page, section, chunk index, text and embedding). Deterministic UUIDv5 identifiers and content-hash checks support idempotent re-runs. Changed documents trigger deletion and rebuilding of their chunks.

For Q&A, a question is embedded using the same model, retrieved via Weaviate vector search, and sent to Ollama with source identifiers. The response is instructed to cite the source identifiers and abstain when evidence is insufficient. For version comparison, full texts of two indexed versions are loaded, split into paragraphs, aligned using SequenceMatcher and diffed deterministically. An LLM classifies the resulting changed blocks into substantive or cosmetic changes with before/after evidence; classification is advisory, not verified fact.

## Feature highlights
Version is a first-class integer property on every chunk, enabling explicit filtering. Comparison operates on both complete document versions rather than unrelated top-k search results. Reruns preserve stable object identities and avoid duplicate chunks.

## Improvements / limitations
PDF reading order, math and tables can be lossy. Heuristic section labels and paragraph alignment may misidentify moved text as new/deleted text. LLM classification can be wrong, and the report caps classifications at 30 changed blocks by default. Add stronger structural PDF parsing, semantic section alignment, evidence verification, citation checking, hybrid search and reranking, golden-set evaluations, ingestion transactions, authentication, and observability for production. Do not claim benchmark results without running them.
