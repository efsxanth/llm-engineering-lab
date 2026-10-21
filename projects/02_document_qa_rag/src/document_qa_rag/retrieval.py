from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore


def retrieve_documents(
    vector_store: VectorStore,
    question: str,
    k: int = 3,
) -> list[Document]:
    """Retrieve documents relevant to a question."""

    if not isinstance(question, str):
        raise TypeError("Question must be a string.")

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer.")

    return vector_store.similarity_search(question, k=k)
