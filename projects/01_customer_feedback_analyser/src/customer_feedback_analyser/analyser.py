import os

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_groq import ChatGroq

from .prompts import customer_feedback_prompt
from .schemas import CustomerFeedback

# Load local environment configuration
load_dotenv()


def analyse_feedback(
    customer_feedback: str, chat: BaseChatModel | None = None
) -> CustomerFeedback:
    """Analyse customer feedback using a hosted LLM."""

    if not isinstance(customer_feedback, str):
        raise TypeError("Customer feedback must be a string.")

    if not customer_feedback.strip():
        raise ValueError("Customer feedback cannot be empty.")

    if chat is None:
        # Retrieve required API configuration
        api_key = os.environ["GROQ_API_KEY"]
        model_name = os.environ["GROQ_MODEL"]
        temperature = float(os.environ["GROQ_TEMPERATURE"])

        chat = ChatGroq(model=model_name, temperature=temperature, api_key=api_key)

    # Define how the LLM response should be parsed
    output_parser = PydanticOutputParser(pydantic_object=CustomerFeedback)

    # Connect prompt, model, and parser using LCEL
    chain = customer_feedback_prompt | chat | output_parser

    # Execute the complete workflow
    result = chain.invoke(
        {
            "customer_feedback": customer_feedback,
            "format_instructions": output_parser.get_format_instructions(),
        }
    )

    return result
