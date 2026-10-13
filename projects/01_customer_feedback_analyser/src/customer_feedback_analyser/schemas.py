from typing import Literal

from pydantic import BaseModel, Field, StrictBool, StrictInt


class CustomerFeedback(BaseModel):
    """
    Structured information extracted from a customer review.
    """

    sentiment: Literal["positive", "neutral", "negative"] = Field(
        description="Overall sentiment expressed in the review."
    )

    is_gift: StrictBool = Field(
        description="Whether the product was purchased as a gift"
    )

    delivery_days: StrictInt = Field(
        description="Number of days delivery took, or -1 if unknown.", ge=-1
    )

    price_assessment: str = Field(
        description="Short summary of the customer's opinion about the price."
    )
