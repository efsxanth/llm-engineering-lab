from unittest.mock import patch

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from document_qa_rag.generation import generate_answer


def test_generate_answer():
    documents = [
        Document(
            page_content=(
                "name: Walking Shoes\n"
                "description: Comfortable shoes with cushioned soles."
            )
        )
    ]

    fake_chat = FakeListChatModel(
        responses=["I recommend Walking Shoes because they have cushioned soles."]
    )

    answer = generate_answer(
        question="What shoes would you recommend?",
        documents=documents,
        chat=fake_chat,
    )

    assert isinstance(answer, str)
    assert "Walking Shoes" in answer
    assert "cushioned soles" in answer


def test_generate_answer_empty_question():
    documents = [Document(page_content="Comfortable walking shoes")]

    fake_chat = FakeListChatModel(responses=["I recommend the walking shoes."])

    with pytest.raises(
        ValueError,
        match="Question cannot be empty",
    ):
        generate_answer(
            question="",
            documents=documents,
            chat=fake_chat,
        )


def test_generate_answer_no_documents():
    fake_chat = FakeListChatModel(responses=["This response should never be used."])

    answer = generate_answer(
        question="Which shoes are waterproof?",
        documents=[],
        chat=fake_chat,
    )

    assert answer == "I cannot answer without supporting documents."


def test_generate_answer_no_documents_skips_llm():
    fake_chat = FakeListChatModel(responses=["This response should never be used."])

    with patch.object(
        FakeListChatModel,
        "invoke",
        side_effect=AssertionError("LLM must not be called"),
    ) as mock_invoke:
        answer = generate_answer(
            question="Which shoes are waterproof?",
            documents=[],
            chat=fake_chat,
        )

    assert answer == "I cannot answer without supporting documents."
    mock_invoke.assert_not_called()
