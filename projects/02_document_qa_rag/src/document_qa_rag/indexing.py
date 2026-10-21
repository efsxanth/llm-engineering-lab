from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore


def build_index(
    documents: list[Document],
    embeddings: Embeddings,
) -> InMemoryVectorStore:
    """Build an in-memory vector index from documents."""

    if not documents:
        raise ValueError("Cannot build an index without documents.")

    vector_store = InMemoryVectorStore(embedding=embeddings)
    vector_store.add_documents(documents)

    return vector_store
