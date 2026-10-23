from collections import defaultdict
from decimal import Decimal
from pathlib import Path

from data_loader import load_sales


def total_revenue(records: list[dict]) -> Decimal:
    """Calculate revenue across all sales records."""

    return sum(
        (record["quantity"] * record["unit_price"] for record in records),
        Decimal(0),
    )


def total_profit(records: list[dict]) -> Decimal:
    """Calculate profit across all sales records."""
    return sum(
        (
            record["quantity"] * (record["unit_price"] - record["unit_cost"])
            for record in records
        ),
        Decimal(0),
    )


def revenue_by_category(records: list[dict]) -> dict[str, Decimal]:
    """Calculate total revenue for each product category."""
    revenue = defaultdict(lambda: Decimal(0))

    for record in records:
        revenue[record["category"]] += record["quantity"] * record["unit_price"]

    return dict(revenue)


if __name__ == "__main__":
    csv_path = Path(__file__).parent / "data" / "sample_sales.csv"
    sales = load_sales(csv_path)

    print("Total revenue:", total_revenue(sales))
    print("Total profit:", total_profit(sales))
    print("Revenue by category:")

    for category, amount in revenue_by_category(sales).items():
        print(f"  {category}: £{amount:.2f}")
