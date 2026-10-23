# LLM Engineering Lab

A collection of hands-on Python projects exploring the design,
development and testing of applications powered by large language
models (LLMs).

The repository focuses on practical LLM engineering using
cloud-hosted models, LangChain and professional Python development
practices.

## Projects

### 01 — Customer Feedback Analyser

An application that transforms customer reviews into structured,
validated data using LangChain, GroqCloud and Pydantic.

Key concepts:
- Prompt templates and structured output parsing.
- LangChain Expression Language (LCEL).
- Pydantic data validation.
- Dependency injection and fake LLM testing.
- Automated testing with pytest.
- Linting and formatting with Ruff.

[View project](projects/01_customer_feedback_analyser/README.md)

### 02 — Document Q&A with Retrieval-Augmented Generation (RAG)

A modular question-answering application that searches an outdoor-product
catalogue using Google Gemini embeddings, then generates answers using
GroqCloud and the retrieved product descriptions. The repository includes
an original, fictional sample catalogue for demonstration.

Key concepts:
- CSV document ingestion and validation.
- Semantic search with embeddings and an in-memory vector store.
- Retrieval-augmented generation with LangChain.
- Source attribution through retrieved documents.
- Retrieval evaluation using Hit Rate@1, Hit Rate@3 and relevance labels.
- Automated testing with pytest and code quality checks with Ruff.

The sample benchmark retrieved a relevant top-ranked product for all eight
labelled questions across 20 products. This is a small demonstration result,
not a general accuracy guarantee.

[View project](projects/02_document_qa_rag/README.md)

### 03 — AI Data Analysis Agent

A Python AI agent built with LangChain, LangGraph and GroqCloud that
analyses a fictional sales dataset using predefined, read-only tools.

Key concepts:
- LLM-driven tool selection and multi-step agent workflows.
- CSV ingestion, validation and monetary calculations using Python.
- Revenue, profit and category-level analysis.
- Dependency injection and offline agent tests using fake LLM responses.
- Agent evaluation of expected tool calls and answer terms.
- Automated testing with pytest and code quality checks with Ruff.

All 11 automated tests passed. In a five-question benchmark using 20 fictional
sales records, the agent matched the expected tool calls and answer terms in
all five cases. These limited checks do not establish general agent accuracy.

[View project](projects/03_data_analysis_agent/README.md)

## Repository Organisation

Each project lives in its own numbered directory under `projects/`
and includes application code, tests and documentation. Some projects
also include example scripts or evaluation datasets.

Each project's README provides setup instructions and commands for running
its application and tests. Real API keys are not stored in the repository;
use the root `.env.example` as a template for a private `.env` file.
