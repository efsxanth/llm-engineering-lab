from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.vectorstores import VectorStore

from document_qa_rag.generation import generate_answer
from document_qa_rag.models import QAResult
from document_qa_rag.retrieval import retrieve_documents


class DocumentQA:
    """Answer questions using an existing vector index."""

    def __init__(
        self,
        vector_store: VectorStore,
        chat: BaseChatModel,
    ) -> None:
        self.vector_store = vector_store
        self.chat = chat

    def ask(self, question: str, k: int = 3) -> str:
        """Retrieve relevant documents and generate an answer."""

        documents = retrieve_documents(
            self.vector_store,
            question,
            k=k,
        )

        return generate_answer(
            question=question,
            documents=documents,
            chat=self.chat,
        )

    def ask_with_sources(self, question: str, k: int = 3) -> QAResult:
        """Answer a question and return the retrieved documents."""

        documents = retrieve_documents(
            self.vector_store,
            question,
            k=k,
        )

        answer = generate_answer(
            question=question,
            documents=documents,
            chat=self.chat,
        )

        return QAResult(
            answer=answer,
            sources=documents,
        )
