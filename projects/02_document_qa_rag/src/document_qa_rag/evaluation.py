import json
from pathlib import Path


def hit_rate_at_1(
    expected_products: list[str],
    retrieved_products: list[str],
) -> float:
    """Calculate the proportion of correct top-ranked products."""

    if not expected_products:
        raise ValueError("Evaluation requires at least one case.")

    if len(expected_products) != len(retrieved_products):
        raise ValueError("Expected and retrieved lists must have equal lengths.")

    correct = sum(
        expected == retrieved
        for expected, retrieved in zip(
            expected_products,
            retrieved_products,
            strict=True,
        )
    )

    return correct / len(expected_products)


def hit_rate_at_k(
    expected_products: list[str],
    retrieved_products: list[list[str]],
    k: int,
) -> float:
    """Calculate how often the expected product appears in the top k."""

    if not expected_products:
        raise ValueError("Evaluation requires at least one case.")

    if len(expected_products) != len(retrieved_products):
        raise ValueError("Expected and retrieved lists must have equal lengths.")

    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer.")

    correct = sum(
        expected in retrieved[:k]
        for expected, retrieved in zip(
            expected_products,
            retrieved_products,
            strict=True,
        )
    )

    return correct / len(expected_products)


def relevance_hit_rate_at_k(
    relevant_products: list[set[str]],
    retrieved_products: list[list[str]],
    k: int,
) -> float:
    """Calculate retrieval hit rate using multiple relevant products."""

    if not relevant_products:
        raise ValueError("Evaluation requires at least one case.")

    if len(relevant_products) != len(retrieved_products):
        raise ValueError("Relevant and retrieved lists must have equal lengths.")

    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer.")

    if any(not products for products in relevant_products):
        raise ValueError("Each case requires at least one relevant product.")

    correct = sum(
        bool(relevant.intersection(retrieved[:k]))
        for relevant, retrieved in zip(
            relevant_products,
            retrieved_products,
            strict=True,
        )
    )

    return correct / len(relevant_products)


def load_evaluation_cases(file_path: str | Path) -> list[dict]:
    """Load and validate retrieval evaluation cases from JSON."""

    with open(file_path, encoding="utf-8") as file:
        cases = json.load(file)

    if not isinstance(cases, list) or not cases:
        raise ValueError("Evaluation data must be a non-empty list.")

    for index, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            raise TypeError(f"Case {index} must be an object.")

        question = case.get("question")
        expected = case.get("expected_product")
        relevant = case.get("relevant_products")

        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Case {index} requires a question.")

        if not isinstance(expected, str) or not expected.strip():
            raise ValueError(f"Case {index} requires an expected product.")

        if (
            not isinstance(relevant, list)
            or not relevant
            or any(not isinstance(name, str) or not name.strip() for name in relevant)
        ):
            raise ValueError(f"Case {index} requires relevant products.")

        if expected not in relevant:
            raise ValueError(f"Case {index}: expected product must be relevant.")

    return cases
