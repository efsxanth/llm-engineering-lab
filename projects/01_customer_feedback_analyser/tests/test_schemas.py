import pytest
from pydantic import ValidationError

from customer_feedback_analyser.schemas import CustomerFeedback


def test_valid_customer_feedback() -> None:
    """A valid customer feedback record should be accepted."""

    feedback = CustomerFeedback(
        sentiment="positive",
        is_gift=True,
        delivery_days=3,
        price_assessment="A little expensive",
    )

    assert feedback.sentiment == "positive"
    assert feedback.is_gift is True
    assert feedback.delivery_days == 3
    assert feedback.price_assessment == "A little expensive"


def test_invalid_sentiment() -> None:
    """An unsupported sentiment should be rejected."""

    with pytest.raises(ValidationError):
        CustomerFeedback(
            sentiment="excellent",
            is_gift=True,
            delivery_days=3,
            price_assessment="A little expensive",
        )


def test_invalid_is_gift() -> None:
    """A string should not be accepted as a Boolean."""

    with pytest.raises(ValidationError):
        CustomerFeedback(
            sentiment="positive",
            is_gift="true",
            delivery_days=3,
            price_assessment="A little expensive.",
        )


def test_invalid_delivery_days() -> None:
    """Delivery days must not be less than -1."""

    with pytest.raises(ValidationError):
        CustomerFeedback(
            sentiment="positive",
            is_gift=True,
            delivery_days=-2,
            price_assessment="A little expensive.",
        )
