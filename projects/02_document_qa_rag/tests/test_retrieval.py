import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from document_qa_rag.indexing import build_index
from document_qa_rag.retrieval import retrieve_documents


def test_retrieve_documents():
    documents = [
        Document(
            page_content="Comfortable walking shoes",
            metadata={"row": 0},
        ),
    ]

    embeddings = DeterministicFakeEmbedding(size=10)
    vector_store = build_index(documents, embeddings)

    results = retrieve_documents(
        vector_store,
        "Comfortable walking shoes",
        k=1,
    )

    assert len(results) == 1
    assert results[0].page_content == "Comfortable walking shoes"
    assert results[0].metadata["row"] == 0


@pytest.mark.parametrize(
    ("question", "expected_exception"),
    [
        ("", ValueError),
        ("   ", ValueError),
        (None, TypeError),
        (123, TypeError),
    ],
)
def test_retrieve_documents_invalid_question(
    question,
    expected_exception,
):
    documents = [
        Document(page_content="Comfortable walking shoes"),
    ]

    embeddings = DeterministicFakeEmbedding(size=10)
    vector_store = build_index(documents, embeddings)

    with pytest.raises(expected_exception):
        retrieve_documents(vector_store, question)


@pytest.mark.parametrize(
    "k",
    [0, -1, 1.5, "3", None, True],
)
def test_retrieve_documents_invalid_k(k):
    documents = [
        Document(page_content="Comfortable walking shoes"),
    ]

    embeddings = DeterministicFakeEmbedding(size=10)
    vector_store = build_index(documents, embeddings)

    with pytest.raises(ValueError, match="k must be a positive integer"):
        retrieve_documents(
            vector_store,
            "Comfortable walking shoes",
            k=k,
        )
