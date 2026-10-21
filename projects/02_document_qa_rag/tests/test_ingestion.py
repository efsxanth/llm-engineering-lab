import pytest

from document_qa_rag.ingestion import load_catalogue


def test_load_catalogue(tmp_path):
    csv_file = tmp_path / "products.csv"

    csv_file.write_text(
        "name,description\nWalking Shoes,Comfortable shoes for daily walking\n",
        encoding="utf-8",
    )

    documents = load_catalogue(str(csv_file))

    assert len(documents) == 1
    assert documents[0].page_content == (
        "name: Walking Shoes\ndescription: Comfortable shoes for daily walking"
    )
    assert documents[0].metadata["row"] == 0
    assert documents[0].metadata["source"] == str(csv_file)


def test_load_catalogue_multiple_rows(tmp_path):
    csv_file = tmp_path / "products.csv"

    csv_file.write_text(
        "name,description\n"
        "Walking Shoes,Comfortable shoes for walking\n"
        "Sun Shirt,Provides UV protection\n",
        encoding="utf-8",
    )

    documents = load_catalogue(str(csv_file))

    assert len(documents) == 2

    assert "Walking Shoes" in documents[0].page_content
    assert "Sun Shirt" in documents[1].page_content

    assert documents[0].metadata["row"] == 0
    assert documents[1].metadata["row"] == 1


def test_load_catalogue_empty_csv(tmp_path):
    csv_file = tmp_path / "empty_products.csv"

    csv_file.write_text(
        "name,description\n",
        encoding="utf-8",
    )

    documents = load_catalogue(str(csv_file))

    assert documents == []


def test_load_catalogue_missing_column(tmp_path):
    csv_file = tmp_path / "invalid_products.csv"

    csv_file.write_text(
        "name,price\nWalking Shoes,49.99\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Missing required columns"):
        load_catalogue(str(csv_file))


def test_load_catalogue_empty_description(tmp_path):
    csv_file = tmp_path / "invalid_products.csv"

    csv_file.write_text(
        "name,description\nSun Shirt,\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Empty product description"):
        load_catalogue(str(csv_file))


def test_load_catalogue_empty_name(tmp_path):
    csv_file = tmp_path / "invalid_products.csv"

    csv_file.write_text(
        "name,description\n,Comfortable shoes for walking\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Empty product name"):
        load_catalogue(str(csv_file))
