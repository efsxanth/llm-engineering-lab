"""Evaluate the live sales agent on a small, synthetic five-question benchmark."""

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from sales_agent import build_agent

CASES_PATH = Path(__file__).resolve().parent / "evaluation" / "agent_cases.json"


def normalise_answer(text: str) -> str:
    """Normalise whitespace, commas and currency symbols for simple checks."""
    return re.sub(r"[,£\s]", "", text).casefold()


def evaluate() -> None:
    load_dotenv()
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    model = ChatGroq(
        model=os.environ["GROQ_MODEL"],
        temperature=0,
        api_key=os.environ["GROQ_API_KEY"],
    )
    agent = build_agent(model)
    tool_passes = 0
    answer_passes = 0
    completed = 0

    for index, case in enumerate(cases, start=1):
        print(f"\n[{index}/{len(cases)}] {case['question']}")
        try:
            result = agent.invoke(
                {"messages": [("human", case["question"])]},
                config={"recursion_limit": 15},
            )
            messages = result["messages"]
            selected_tools = [
                call["name"]
                for message in messages
                for call in getattr(message, "tool_calls", [])
            ]
            answer = str(messages[-1].content)
            expected_tools = case["expected_tools"]
            # Compare tools without order; repeated or unexpected calls still fail.
            tools_ok = sorted(selected_tools) == sorted(expected_tools)
            normalised = normalise_answer(answer)
            answers_ok = all(
                normalise_answer(term) in normalised
                for term in case["expected_answer_terms"]
            )
            completed += 1
            tool_passes += tools_ok
            answer_passes += answers_ok
            print("Tools used:", selected_tools)
            print("Tool selection:", "PASS" if tools_ok else "FAIL")
            print("Answer check:", "PASS" if answers_ok else "FAIL")
            print("Final answer:", answer)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: {type(exc).__name__}: {exc}")

    total = len(cases)
    print("\n--- Evaluation summary ---")
    print(f"Completed: {completed}/{total}")
    print(f"Exact expected tool-call matches: {tool_passes}/{total}")
    print(f"Answer term checks passed: {answer_passes}/{total}")
    print("Benchmark: five questions on 20 fictional sales records.")
    print("Answer checks are simple term matching, not factual verification.")


if __name__ == "__main__":
    evaluate()
