import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from document_qa_rag.evaluation import (
    hit_rate_at_1,
    hit_rate_at_k,
    load_evaluation_cases,
    relevance_hit_rate_at_k,
)
from document_qa_rag.indexing import build_index
from document_qa_rag.ingestion import load_catalogue
from document_qa_rag.retrieval import retrieve_documents

SAMPLE_SIZE = 20


def main() -> None:
    load_dotenv()

    project_dir = Path(__file__).resolve().parents[1]
    catalogue_path = project_dir / "data" / "sample_outdoor_catalogue.csv"
    evaluation_path = project_dir / "evaluation" / "retrieval_cases.json"
    evaluation_cases = load_evaluation_cases(evaluation_path)

    documents = load_catalogue(str(catalogue_path))
    sample_documents = documents[:SAMPLE_SIZE]
    print(f"Evaluating retrieval across {len(sample_documents)} products.")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=os.environ["GEMINI_API_KEY"],
    )

    vector_store = build_index(sample_documents, embeddings)

    expected_products = []
    relevant_products = []
    top_one_products = []
    top_three_products = []

    for case in evaluation_cases:
        results = retrieve_documents(
            vector_store,
            case["question"],
            k=3,
        )

        retrieved_names = [
            document.page_content.splitlines()[0] for document in results
        ]

        expected_name = f"name: {case['expected_product']}"

        expected_products.append(expected_name)
        relevant_products.append(
            {f"name: {name}" for name in case["relevant_products"]}
        )
        top_one_products.append(retrieved_names[0])
        top_three_products.append(retrieved_names)

        print(f"\nQuestion: {case['question']}")
        print(f"Expected: {expected_name}")

        for rank, name in enumerate(retrieved_names, start=1):
            print(f"{rank}. {name}")

        print(f"Hit@1: {expected_name == retrieved_names[0]}")
        print(f"Hit@3: {expected_name in retrieved_names}")
        print(f"Relevant Hit@1: {retrieved_names[0] in relevant_products[-1]}")
        print(
            "Relevant Hit@3: "
            f"{bool(relevant_products[-1].intersection(retrieved_names))}"
        )

    hit_rate_1 = hit_rate_at_1(
        expected_products,
        top_one_products,
    )

    hit_rate_3 = hit_rate_at_k(
        expected_products,
        top_three_products,
        k=3,
    )

    relevant_hit_rate_1 = relevance_hit_rate_at_k(
        relevant_products,
        top_three_products,
        k=1,
    )

    relevant_hit_rate_3 = relevance_hit_rate_at_k(
        relevant_products,
        top_three_products,
        k=3,
    )

    print(f"\nRetrieval Hit Rate@1: {hit_rate_1:.1%}")
    print(f"Retrieval Hit Rate@3: {hit_rate_3:.1%}")
    print(f"Relevance-aware Hit Rate@1: {relevant_hit_rate_1:.1%}")
    print(f"Relevance-aware Hit Rate@3: {relevant_hit_rate_3:.1%}")


if __name__ == "__main__":
    main()
