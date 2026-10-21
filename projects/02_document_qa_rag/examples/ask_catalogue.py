import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq

from document_qa_rag.application import DocumentQA
from document_qa_rag.indexing import build_index
from document_qa_rag.ingestion import load_catalogue


def main() -> None:
    load_dotenv()

    project_dir = Path(__file__).resolve().parents[1]
    catalogue_path = project_dir / "data" / "sample_outdoor_catalogue.csv"

    documents = load_catalogue(str(catalogue_path))

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=os.environ["GEMINI_API_KEY"],
    )
    vector_store = build_index(documents, embeddings)

    chat = ChatGroq(
        model=os.environ["GROQ_MODEL"],
        temperature=float(os.environ["GROQ_TEMPERATURE"]),
        api_key=os.environ["GROQ_API_KEY"],
    )

    app = DocumentQA(vector_store=vector_store, chat=chat)
    question = "I need comfortable shoes for short everyday walks. What would you recommend?"
    result = app.ask_with_sources(question, k=3)

    print("\nQuestion:", question)
    print("\nRAG answer:\n", result.answer)
    print("\nRetrieved sources:")
    for number, document in enumerate(result.sources, start=1):
        print(f"\n--- Source {number} ---")
        print(document.page_content)
        print("Metadata:", document.metadata)


if __name__ == "__main__":
    main()
