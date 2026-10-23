import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from data_loader import load_sales


def test_load_valid_sales():
    path = PROJECT_DIR / "data" / "sample_sales.csv"

    records = load_sales(path)

    assert len(records) == 20
    assert records[0]["order_id"] == "S001"


def test_missing_required_column(tmp_path):
    path = tmp_path / "missing_column.csv"
    path.write_text(
        "order_id,quantity,unit_price\nS001,3,65.00\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Missing columns"):
        load_sales(path)


def test_invalid_quantity(tmp_path):
    path = tmp_path / "invalid_quantity.csv"
    path.write_text(
        "order_id,order_date,category,product,"
        "quantity,unit_price,unit_cost\n"
        "S001,2026-09-01,Footwear,Shoes,abc,65.00,38.00\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid numerical data"):
        load_sales(path)


def test_negative_price(tmp_path):
    path = tmp_path / "negative_price.csv"
    path.write_text(
        "order_id,order_date,category,product,"
        "quantity,unit_price,unit_cost\n"
        "S001,2026-09-01,Footwear,Shoes,3,-65.00,38.00\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid numerical data"):
        load_sales(path)


def test_empty_csv(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text(
        "order_id,order_date,category,product,quantity,unit_price,unit_cost\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no sales records"):
        load_sales(path)
