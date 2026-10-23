import csv
from decimal import Decimal, InvalidOperation
from pathlib import Path

REQUIRED_COLUMNS = {
    "order_id",
    "order_date",
    "category",
    "product",
    "quantity",
    "unit_price",
    "unit_cost",
}


def load_sales(csv_path: Path) -> list[dict]:
    """Load and validate sales records from a CSV file."""

    with csv_path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError("CSV file is empty or missing a header.")

        missing = REQUIRED_COLUMNS - set(reader.fieldnames)

        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")

        records = []

        for row_number, row in enumerate(reader, start=2):
            try:
                quantity = int(row["quantity"])
                unit_price = Decimal(row["unit_price"])
                unit_cost = Decimal(row["unit_cost"])

                if quantity <= 0 or unit_price < 0 or unit_cost < 0:
                    raise ValueError("Negative values or invalid quantity.")

            except (ValueError, InvalidOperation, TypeError) as exc:
                raise ValueError(f"Invalid numerical data on row {row_number}") from exc

            records.append(
                {
                    **row,
                    "quantity": quantity,
                    "unit_price": unit_price,
                    "unit_cost": unit_cost,
                }
            )

    if not records:
        raise ValueError("CSV file contains no sales records.")

    return records


if __name__ == "__main__":
    dataset_path = Path(__file__).parent / "data" / "sample_sales.csv"

    sales = load_sales(dataset_path)

    print("Records loaded:", len(sales))
    print("First record:", sales[0])
