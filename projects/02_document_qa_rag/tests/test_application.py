from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from document_qa_rag.application import DocumentQA
from document_qa_rag.indexing import build_index
from document_qa_rag.models import QAResult


def test_document_qa_ask():
    documents = [
        Document(
            page_content=(
                "name: Walking Shoes\n"
                "description: Comfortable shoes with cushioned soles."
            ),
            metadata={"row": 0},
        )
    ]

    embeddings = DeterministicFakeEmbedding(size=10)
    vector_store = build_index(documents, embeddings)

    fake_chat = FakeListChatModel(
        responses=["I recommend Walking Shoes because they have cushioned soles."]
    )

    app = DocumentQA(
        vector_store=vector_store,
        chat=fake_chat,
    )

    answer = app.ask(
        "What shoes would you recommend?",
        k=1,
    )

    assert isinstance(answer, str)
    assert "Walking Shoes" in answer
    assert "cushioned soles" in answer


def test_document_qa_multiple_questions():
    documents = [
        Document(
            page_content="name: Walking Shoes\n"
            "description: Comfortable shoes with cushioned soles.",
        ),
    ]

    embeddings = DeterministicFakeEmbedding(size=10)
    vector_store = build_index(documents, embeddings)

    fake_chat = FakeListChatModel(
        responses=[
            "Walking Shoes have cushioned soles.",
            "Walking Shoes are designed for comfort.",
        ]
    )

    app = DocumentQA(
        vector_store=vector_store,
        chat=fake_chat,
    )

    first_answer = app.ask("Which shoes have cushioning?", k=1)
    second_answer = app.ask("Which shoes are comfortable?", k=1)

    assert "cushioned soles" in first_answer
    assert "designed for comfort" in second_answer


def test_document_qa_ask_with_sources():
    documents = [
        Document(
            page_content=(
                "name: Walking Shoes\n"
                "description: Comfortable shoes with cushioned soles."
            ),
            metadata={"row": 0},
        )
    ]

    embeddings = DeterministicFakeEmbedding(size=10)
    vector_store = build_index(documents, embeddings)

    fake_chat = FakeListChatModel(responses=["Walking Shoes have cushioned soles."])

    app = DocumentQA(
        vector_store=vector_store,
        chat=fake_chat,
    )

    result = app.ask_with_sources(
        "Which shoes are comfortable?",
        k=1,
    )

    assert isinstance(result, QAResult)
    assert result.answer == "Walking Shoes have cushioned soles."
    assert len(result.sources) == 1
    assert result.sources[0].page_content == documents[0].page_content
    assert result.sources[0].metadata["row"] == 0
