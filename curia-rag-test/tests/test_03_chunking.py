"""Step 3 — Chunking: splits correctly, overlap preserved, parent_id tracked."""
import pytest
from datetime import date
from src.models import Document
from src.chunking import chunk_document, chunk_documents, tokenize


def _doc(text: str, doc_id: str = "d1") -> Document:
    return Document(id=doc_id, title="T", source="job_posting",
                    date=date(2025, 1, 1), text=text)


def test_empty_document_returns_no_chunks():
    assert chunk_document(_doc("")) == []


def test_short_document_returns_one_chunk():
    chunks = chunk_document(_doc("hello world foo bar"), max_tokens=50)
    assert len(chunks) == 1


def test_long_document_splits_into_multiple_chunks():
    long_text = " ".join([f"word{i}" for i in range(500)])
    chunks = chunk_document(_doc(long_text), max_tokens=100, overlap=20)
    assert len(chunks) > 1


def test_all_chunks_inherit_parent_id():
    long_text = " ".join([f"word{i}" for i in range(500)])
    chunks = chunk_document(_doc(long_text, doc_id="myparent"), max_tokens=100, overlap=20)
    for chunk in chunks:
        assert chunk.parent_id == "myparent"
        assert chunk.chunk_id.startswith("myparent_c")


def test_chunk_ids_are_unique():
    long_text = " ".join([f"word{i}" for i in range(500)])
    chunks = chunk_document(_doc(long_text), max_tokens=100, overlap=20)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_indices_are_sequential():
    long_text = " ".join([f"word{i}" for i in range(500)])
    chunks = chunk_document(_doc(long_text), max_tokens=100, overlap=20)
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i


def test_overlap_creates_shared_content():
    """With overlap, the end of chunk N should appear at start of chunk N+1."""
    words = [f"w{i}" for i in range(60)]
    text = " ".join(words)
    chunks = chunk_document(_doc(text), max_tokens=30, overlap=10)
    if len(chunks) >= 2:
        tokens_0 = tokenize(chunks[0].text)
        tokens_1 = tokenize(chunks[1].text)
        overlap_tokens = set(tokens_0[-10:]) & set(tokens_1[:10])
        assert len(overlap_tokens) > 0


def test_chunk_documents_batches_multiple_docs():
    docs = [_doc(f"document {i} " + " ".join([f"w{j}" for j in range(50)]),
                 doc_id=f"doc_{i}")
            for i in range(3)]
    chunks = chunk_documents(docs, max_tokens=30, overlap=5)
    parent_ids = {c.parent_id for c in chunks}
    assert parent_ids == {"doc_0", "doc_1", "doc_2"}


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        chunk_document(_doc("x"), max_tokens=0)
    with pytest.raises(ValueError):
        chunk_document(_doc("x"), max_tokens=10, overlap=10)


def test_metadata_preserved_in_chunks():
    doc = Document(id="d1", title="T", source="job_posting",
                   date=date(2025, 1, 1), text="hello " * 200,
                   metadata={"company": "acme"})
    chunks = chunk_document(doc, max_tokens=50)
    for chunk in chunks:
        assert chunk.metadata.get("company") == "acme"
