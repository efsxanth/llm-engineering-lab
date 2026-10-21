import csv

from langchain_core.documents import Document


def load_catalogue(file_path: str) -> list[Document]:
    """Load product records from a CSV file."""

    documents = []

    with open(file_path, encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)

        required_columns = {"name", "description"}
        actual_columns = set(reader.fieldnames or [])
        missing_columns = required_columns - actual_columns

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {', '.join(sorted(missing_columns))}"
            )

        for row_number, row in enumerate(reader):
            name = row["name"]
            description = row["description"]

            if not description or not description.strip():
                raise ValueError(
                    f"Empty product description at CSV row {row_number + 2}"
                )

            if not name or not name.strip():
                raise ValueError(f"Empty product name at CSV row {row_number + 2}")

            document = Document(
                page_content=(f"name: {name}\ndescription: {description}"),
                metadata={
                    "source": file_path,
                    "row": row_number,
                },
            )

            documents.append(document)

    return documents
