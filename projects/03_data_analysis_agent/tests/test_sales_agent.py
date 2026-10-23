import sys
from pathlib import Path

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from sales_agent import build_agent


class FakeToolCallingModel(FakeMessagesListChatModel):
    """Fake LLM that supports binding tools."""

    def bind_tools(self, tools, **kwargs):
        return self


def test_agent_selects_revenue_by_category():
    fake_model = FakeToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_revenue_by_category",
                        "args": {},
                        "id": "test-call-1",
                    }
                ],
            ),
            AIMessage(content=("Footwear generated the highest revenue at £1,210.00.")),
        ]
    )

    agent = build_agent(fake_model)

    result = agent.invoke(
        {
            "messages": [
                (
                    "human",
                    "Which category generated the highest revenue?",
                )
            ]
        },
        config={"recursion_limit": 10},
    )

    messages = result["messages"]

    assert messages[1].tool_calls[0]["name"] == ("get_revenue_by_category")

    assert isinstance(messages[2], ToolMessage)
    assert '"Footwear": "£1,210.00"' in messages[2].content

    assert "Footwear" in messages[-1].content
    assert "£1,210.00" in messages[-1].content


def test_agent_executes_multiple_tools():
    fake_model = FakeToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_total_revenue",
                        "args": {},
                        "id": "revenue-call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_total_profit",
                        "args": {},
                        "id": "profit-call",
                    }
                ],
            ),
            AIMessage(
                content=("Total revenue is £3,678.50 and total profit is £1,609.50.")
            ),
        ]
    )

    agent = build_agent(fake_model)

    result = agent.invoke(
        {"messages": [("human", "What are total revenue and total profit?")]},
        config={"recursion_limit": 10},
    )

    tool_messages = [
        message for message in result["messages"] if isinstance(message, ToolMessage)
    ]

    assert len(tool_messages) == 2
    assert tool_messages[0].content == "£3,678.50"
    assert tool_messages[1].content == "£1,609.50"

    final_answer = result["messages"][-1].content

    assert "£3,678.50" in final_answer
    assert "£1,609.50" in final_answer
