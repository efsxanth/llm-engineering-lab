import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from document_qa_rag.indexing import build_index


def test_build_index():
    documents = [
        Document(
            page_content="Comfortable walking shoes",
            metadata={"row": 0},
        ),
    ]

    embeddings = DeterministicFakeEmbedding(size=10)

    vector_store = build_index(documents, embeddings)

    results = vector_store.similarity_search(
        "Comfortable walking shoes",
        k=1,
    )

    assert len(results) == 1
    assert results[0].page_content == "Comfortable walking shoes"
    assert results[0].metadata["row"] == 0


def test_build_index_empty_documents():
    embeddings = DeterministicFakeEmbedding(size=10)

    with pytest.raises(ValueError, match="Cannot build an index without documents"):
        build_index([], embeddings)
