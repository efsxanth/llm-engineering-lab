# Document Q&A with Retrieval-Augmented Generation (RAG)

A modular Python application that answers questions about an outdoor-clothing catalogue using retrieved product descriptions. Built as part of the [LLM Engineering Lab](../../README.md).

## Overview

The application combines **Google Gemini embeddings** for semantic retrieval with **GroqCloud** for answer generation. It uses LangChain's in-memory vector store and returns the documents retrieved to support an answer.

```text
Outdoor-clothing CSV
    -> ingestion.py (validate and load product documents)
    -> indexing.py (Gemini embeddings; in-memory vector index)
User question
    -> retrieval.py (semantic similarity search)
    -> generation.py (LangChain prompt + Groq chat model)
    -> application.py (DocumentQA orchestration)
    -> QAResult (answer + retrieved source documents)
```

## Features

- Loads and validates product names and descriptions from CSV.
- Builds a semantic index from document embeddings.
- Retrieves the top `k` relevant catalogue documents.
- Generates recommendations constrained by retrieved descriptions.
- Supports both `ask()` (answer text) and `ask_with_sources()` (answer plus source documents).
- Includes an independently testable JSON evaluation-data loader and retrieval metrics.
- Uses dependency injection and fake models for API-free tests.

## Requirements

- Python 3.12 or newer
- Gemini API key (`GEMINI_API_KEY`) for embeddings
- GroqCloud API key (`GROQ_API_KEY`) for answer generation
- Network access for the live examples
- The included `data/sample_outdoor_catalogue.csv` for the demonstration scripts

## Setup (Windows PowerShell)

From the **repository root**:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".\projects\02_document_qa_rag[dev]"
```

Add the following to your local repository-root `.env` file (never commit credentials):

```dotenv
GEMINI_API_KEY=replace_with_your_gemini_key
GROQ_API_KEY=replace_with_your_groq_key
GROQ_MODEL=openai/gpt-oss-20b
GROQ_TEMPERATURE=0
```

`python-dotenv` loads `.env` in the examples. A safe placeholder template should be available at the repository root as `.env.example`.

## Dataset

This repository includes an **original, fictional 20-product sample catalogue**:

```text
projects/02_document_qa_rag/data/sample_outdoor_catalogue.csv
```

The CSV contains `name` and `description` columns and is suitable for sharing publicly. It demonstrates product retrieval without relying on redistribution of the DeepLearning.AI course dataset.

**Important:** The example scripts and evaluation labels originally used `OutdoorClothingCatalog_1000.csv`. Before running the live examples with this sample, update `ask_catalogue.py` and `evaluate_retrieval.py` to load `sample_outdoor_catalogue.csv`, and update `evaluation/retrieval_cases.json` to name products in the new catalogue. The offline tests do not require the course dataset.

## Run the application

From the repository root:

```powershell
python .\projects\02_document_qa_rag\examples\ask_catalogue.py
```

Once updated to use the included sample CSV, this example will index the fictional catalogue, answer a sample question and print retrieved source documents with metadata. It calls external embedding and chat APIs. Until the script has been updated, it may still reference the old course CSV.

## Retrieval evaluation

```powershell
python .\projects\02_document_qa_rag\examples\evaluate_retrieval.py
```

The benchmark reads `evaluation/retrieval_cases.json` and reports strict and relevance-aware Hit Rate@1/@3 scores. It calls Gemini for embeddings but does not call GroqCloud. Update the benchmark CSV path and relevance labels to use the included fictional catalogue before running it.

Historical results using the **original course catalogue** (six manually labelled questions, 50 indexed products; **not results for the new sample catalogue**):

| Metric | Observed score |
| --- | ---: |
| Strict Hit Rate@1 | 66.7% |
| Strict Hit Rate@3 | 83.3% |
| Relevance-aware Hit Rate@1 | 83.3% |
| Relevance-aware Hit Rate@3 | 100.0% |

These results are **illustrative, not a general accuracy guarantee**: only six questions were evaluated and relevance labels may be incomplete. A retrieved document can be relevant while an LLM-generated claim remains unsupported.

## Tests and code quality

Run offline tests from the repository root:

```powershell
python -m pytest .\projects\02_document_qa_rag\tests -v
python -m ruff check .\projects\02_document_qa_rag
python -m ruff format --check .\projects\02_document_qa_rag
```

**Latest user-reported verification:** 43 tests passed with clean Ruff checks. No external API calls are required for the unit/integration test suite.

## Package organisation

```text
02_document_qa_rag/
├── pyproject.toml
├── README.md
├── data/
│   └── sample_outdoor_catalogue.csv  # Original, fictional demo data
├── evaluation/
│   └── retrieval_cases.json
├── examples/
│   ├── ask_catalogue.py
│   └── evaluate_retrieval.py
├── src/document_qa_rag/
│   ├── __init__.py
│   ├── application.py
│   ├── evaluation.py
│   ├── generation.py
│   ├── indexing.py
│   ├── ingestion.py
│   ├── models.py
│   └── retrieval.py
└── tests/                         # pytest suite
```

## Limitations and future improvements

- The vector store is **in-memory**: document embeddings are regenerated when the script restarts.
- Semantic similarity may rank a related product above one matching an exact technical specification.
- Source documents are returned for inspection, but **source attribution is not automatic factual verification**.
- The evaluation labels must be revised for the fictional sample catalogue; historical scores do not transfer to a different dataset.
- The live benchmark uses a small, non-exhaustive set of manually labelled cases.
- Optional future work: persistent indexing, hybrid keyword/semantic search, broader evaluation cases, and claim-level answer-groundedness checks.

## Security

Do not commit `.env`, live credentials, virtual environments, or personal/local workspaces. Review `git status` and staged changes before pushing.
