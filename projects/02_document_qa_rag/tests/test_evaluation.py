import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from document_qa_rag.evaluation import (
    hit_rate_at_1,
    hit_rate_at_k,
    load_evaluation_cases,
    relevance_hit_rate_at_k,
)
from document_qa_rag.indexing import build_index
from document_qa_rag.retrieval import retrieve_documents


def test_retrieval_returns_relevant_product():
    documents = [
        Document(
            page_content=(
                "name: Walking Shoes\n"
                "description: Comfortable shoes with cushioned soles."
            )
        )
    ]

    embeddings = DeterministicFakeEmbedding(size=10)
    vector_store = build_index(documents, embeddings)

    results = retrieve_documents(
        vector_store,
        "Which shoes are comfortable?",
        k=1,
    )

    assert len(results) == 1
    assert "Walking Shoes" in results[0].page_content


def test_hit_rate_at_1_all_correct():
    result = hit_rate_at_1(
        expected_products=["Shoes", "Pants"],
        retrieved_products=["Shoes", "Pants"],
    )

    assert result == 1.0


def test_hit_rate_at_1_one_correct():
    result = hit_rate_at_1(
        expected_products=["Shoes", "Pants"],
        retrieved_products=["Shoes", "Jacket"],
    )

    assert result == 0.5


def test_hit_rate_at_1_none_correct():
    result = hit_rate_at_1(
        expected_products=["Shoes", "Pants"],
        retrieved_products=["Jacket", "Shoes"],
    )

    assert result == 0.0


def test_hit_rate_at_1_empty_dataset():
    with pytest.raises(ValueError, match="at least one case"):
        hit_rate_at_1([], [])


def test_hit_rate_at_1_mismatched_lengths():
    with pytest.raises(ValueError, match="equal lengths"):
        hit_rate_at_1(["Shoes", "Pants"], ["Shoes"])


def test_hit_rate_at_k_all_correct():
    result = hit_rate_at_k(
        expected_products=["Shoes", "Pants"],
        retrieved_products=[
            ["Boots", "Shoes", "Jacket"],
            ["Pants", "Jacket", "Boots"],
        ],
        k=3,
    )

    assert result == 1.0


def test_hit_rate_at_k_partially_correct():
    result = hit_rate_at_k(
        expected_products=["Shoes", "Pants"],
        retrieved_products=[
            ["Boots", "Shoes", "Jacket"],
            ["Jacket", "Boots", "Hat"],
        ],
        k=2,
    )

    assert result == 0.5


def test_hit_rate_at_k_respects_cutoff():
    result = hit_rate_at_k(
        expected_products=["Shoes"],
        retrieved_products=[["Boots", "Jacket", "Shoes"]],
        k=2,
    )

    assert result == 0.0


def test_hit_rate_at_k_invalid_k():
    with pytest.raises(ValueError, match="positive integer"):
        hit_rate_at_k(["Shoes"], [["Shoes"]], k=0)


def test_hit_rate_at_k_empty_dataset():
    with pytest.raises(ValueError, match="at least one case"):
        hit_rate_at_k([], [], k=3)


def test_hit_rate_at_k_mismatched_lengths():
    with pytest.raises(ValueError, match="equal lengths"):
        hit_rate_at_k(
            ["Shoes", "Pants"],
            [["Shoes"]],
            k=3,
        )


def test_relevance_hit_rate_at_k_multiple_relevant():
    result = relevance_hit_rate_at_k(
        relevant_products=[
            {"Walking Shoes", "Running Shoes"},
            {"Waterproof Pants"},
        ],
        retrieved_products=[
            ["Running Shoes", "Boots"],
            ["Jacket", "Waterproof Pants"],
        ],
        k=2,
    )

    assert result == 1.0


def test_relevance_hit_rate_at_k_partial_success():
    result = relevance_hit_rate_at_k(
        relevant_products=[
            {"Walking Shoes", "Running Shoes"},
            {"Waterproof Pants"},
        ],
        retrieved_products=[
            ["Running Shoes"],
            ["Jacket"],
        ],
        k=1,
    )

    assert result == 0.5


def test_relevance_hit_rate_at_k_empty_relevant_set():
    with pytest.raises(ValueError, match="at least one relevant product"):
        relevance_hit_rate_at_k(
            relevant_products=[set()],
            retrieved_products=[["Walking Shoes"]],
            k=1,
        )


def test_load_evaluation_cases_valid(tmp_path):
    file_path = tmp_path / "cases.json"
    file_path.write_text(
        '[{"question": "Which shoes are comfortable?", '
        '"expected_product": "Walking Shoes", '
        '"relevant_products": ["Walking Shoes"]}]',
        encoding="utf-8",
    )

    cases = load_evaluation_cases(file_path)

    assert len(cases) == 1
    assert cases[0]["expected_product"] == "Walking Shoes"


def test_load_evaluation_cases_missing_question(tmp_path):
    file_path = tmp_path / "cases.json"
    file_path.write_text(
        '[{"expected_product": "Walking Shoes", '
        '"relevant_products": ["Walking Shoes"]}]',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires a question"):
        load_evaluation_cases(file_path)
