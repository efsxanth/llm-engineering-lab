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

## Repository Organisation

Each project lives in its own numbered directory under `projects/`
and contains its own source code, examples, tests and documentation.

Each project's README provides installation instructions, API configuration,
and commands for running its examples and tests. Real API keys are not stored
in the repository; use `.env.example` as a template for a private `.env` file.

Additional projects may be added as the lab develops.

### 03 — AI Data Analysis Agent

A Python AI agent built with **LangChain, LangGraph and GroqCloud** that analyses synthetic sales data using predefined analytical tools.

The agent autonomously selects tools to calculate revenue, profit and category-level statistics, then generates answers using the calculated results.

Includes CSV validation, multi-step tool execution, **11 automated tests** and a five-question evaluation benchmark. All five cases passed the expected tool-selection and answer checks.

[View Project 03](projects/03_data_analysis_agent/README.md)