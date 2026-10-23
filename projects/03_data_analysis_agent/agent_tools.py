from pathlib import Path

from analytics import (
    revenue_by_category,
    total_profit,
    total_revenue,
)
from data_loader import load_sales
from langchain_core.tools import tool

DATA_PATH = Path(__file__).parent / "data" / "sample_sales.csv"


@tool
def get_total_revenue() -> str:
    """Calculate total sales revenue across all orders."""
    records = load_sales(DATA_PATH)
    return f"£{total_revenue(records):,.2f}"


@tool
def get_total_profit() -> str:
    """Calculate total profit across all orders."""
    records = load_sales(DATA_PATH)
    return f"£{total_profit(records):,.2f}"


@tool
def get_revenue_by_category() -> dict[str, str]:
    """Calculate sales revenue for each product category."""
    records = load_sales(DATA_PATH)
    results = revenue_by_category(records)

    return {category: f"£{amount:,.2f}" for category, amount in results.items()}


if __name__ == "__main__":
    print(get_total_revenue.invoke({}))
    print(get_total_profit.invoke({}))
    print(get_revenue_by_category.invoke({}))
