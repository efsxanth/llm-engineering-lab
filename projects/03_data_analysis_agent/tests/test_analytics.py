import sys
from decimal import Decimal
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from analytics import revenue_by_category, total_profit, total_revenue
from data_loader import load_sales


@pytest.fixture
def sales_records():
    csv_path = PROJECT_DIR / "data" / "sample_sales.csv"
    return load_sales(csv_path)


def test_total_revenue(sales_records):
    assert total_revenue(sales_records) == Decimal("3678.50")


def test_total_profit(sales_records):
    assert total_profit(sales_records) == Decimal("1609.50")


def test_revenue_by_category(sales_records):
    results = revenue_by_category(sales_records)

    assert results == {
        "Footwear": Decimal("1210.00"),
        "Outerwear": Decimal("1182.00"),
        "Accessories": Decimal("521.50"),
        "Equipment": Decimal("765.00"),
    }


def test_category_revenue_matches_total(sales_records):
    category_totals = revenue_by_category(sales_records)

    assert sum(category_totals.values()) == total_revenue(sales_records)
