import json

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from customer_feedback_analyser.analyser import analyse_feedback


def test_analyse_feedback_with_fake_llm() -> None:
    """Analyse feedback without calling the real Groq API."""

    fake_response = {
        "sentiment": "positive",
        "is_gift": True,
        "delivery_days": 3,
        "price_assessment": "A little expensive.",
    }

    fake_chat = FakeListChatModel(responses=[json.dumps(fake_response)])

    result = analyse_feedback("I bought these headphones as a gift.", chat=fake_chat)

    assert result.sentiment == "positive"
    assert result.is_gift is True
    assert result.delivery_days == 3
    assert result.price_assessment == "A little expensive."


def test_analyse_feedback_with_invalid_llm_response() -> None:
    """Rejects an LLM response that violates the schema."""

    fake_response = {
        "sentiment": "excellent",
        "is_gift": True,
        "delivery_days": 3,
        "price_assessment": "A little expensive.",
    }

    fake_chat = FakeListChatModel(responses=[json.dumps(fake_response)])

    with pytest.raises(OutputParserException):
        analyse_feedback("I bought these headphones as a gift.", chat=fake_chat)


@pytest.mark.parametrize("customer_feedback", ["", "   ", "\n\t"])
def test_analyse_feedback_rejects_empty_inputs(customer_feedback: str) -> None:
    """Rejects empty or whitespace-only customer feedback."""

    with pytest.raises(ValueError):
        analyse_feedback(customer_feedback)


@pytest.mark.parametrize("customer_feedback", [123, None, True])
def test_analyse_feedback_rejects_non_string_input(customer_feedback: object) -> None:
    """Rejects customer feedback that is not a string."""

    with pytest.raises(TypeError):
        analyse_feedback(customer_feedback)
