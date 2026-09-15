"""JARVIS RAG layer — retrieval-augmented answering over the user's own files.

Everything here is local: no embeddings service, no API key, no downloaded
model. The retriever is a hand-built BM25 inverted index plus the character
n-gram vectoriser we already use in ``brain/knowledge.py``, and the answer
composer is *extractive* — it can only repeat sentences that literally exist in
an indexed file, which is what makes the citations trustworthy.

Sub-modules
    extract   file discovery, encoding repair, binary rejection, path policy
    chunk     Hebrew-aware chunking with exact line spans (for citations)
    index     SQLite index: docs, chunks, inverted postings, incremental sync
    retrieve  hybrid BM25 + n-gram scoring, phrase bonus, MMR diversity
    answer    grounded extractive composer with source citations
    engine    the facade the skills, the router and the server all talk to
"""

from __future__ import annotations

from brain.rag.answer import GroundedAnswer, compose_answer  # noqa: F401
from brain.rag.chunk import Chunk, chunk_text  # noqa: F401
from brain.rag.engine import RAG, RagEngine  # noqa: F401
from brain.rag.extract import ExtractResult, decode_text, extract_file, walk_files  # noqa: F401
from brain.rag.index import RagIndex  # noqa: F401
from brain.rag.retrieve import Hit, Retriever  # noqa: F401

__all__ = [
    "Chunk", "chunk_text",
    "ExtractResult", "decode_text", "extract_file", "walk_files",
    "RagIndex", "Retriever", "Hit",
    "GroundedAnswer", "compose_answer",
    "RagEngine", "RAG",
]
