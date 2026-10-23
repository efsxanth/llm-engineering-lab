import os

from agent_tools import (
    get_revenue_by_category,
    get_total_profit,
    get_total_revenue,
)
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq

TOOLS = [
    get_total_revenue,
    get_total_profit,
    get_revenue_by_category,
]

SYSTEM_PROMPT = (
    "You are a sales data analysis assistant. "
    "Use the provided tools to answer questions about sales. "
    "Do not invent numerical results. "
    "Do not claim to have analysed data unless a tool "
    "provided the relevant result."
)


def build_agent(model: BaseChatModel):
    """Create the sales agent using the supplied language model."""
    return create_agent(
        model=model,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
    )


def main() -> None:
    load_dotenv()

    model = ChatGroq(
        model=os.environ["GROQ_MODEL"],
        temperature=0,
        api_key=os.environ["GROQ_API_KEY"],
    )

    agent = build_agent(model)

    question = "Which product category generated the highest revenue?"

    result = agent.invoke(
        {"messages": [("human", question)]},
        config={"recursion_limit": 10},
    )

    print("Question:", question)
    print("\nAgent execution:\n")

    for message in result["messages"]:
        if getattr(message, "tool_calls", None):
            print("Requested tools:", message.tool_calls)

        if message.type == "tool":
            print("Tool result:", message.content)

    print("\nFinal answer:")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
