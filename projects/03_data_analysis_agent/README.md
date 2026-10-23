# Project 03 — AI Data Analysis Agent

A Python AI agent that analyses a synthetic sales dataset using LangChain, LangGraph and GroqCloud.

The agent selects appropriate analytical tools based on a user's question, executes calculations using Python and generates answers grounded in the tool results.

## Features

- AI agent built with LangChain's `create_agent()` interface, powered by LangGraph.
- GroqCloud LLM (`openai/gpt-oss-20b`) for tool selection and answer generation.
- Three analytical tools for total revenue, total profit and revenue by category.
- CSV loading and validation with Python's standard library.
- Decimal arithmetic for monetary calculations.
- Automated tests, including offline agent tests using fake LLM responses.
- Evaluation of tool selection and final answers against a small benchmark.

## How it works

The application follows this workflow:

1. The user submits a sales question.
2. The LLM selects an appropriate analytical tool.
3. LangChain executes the Python function.
4. The tool returns a calculated result.
5. The agent uses the result to generate its answer.

The agent can make multiple tool calls when a question requires more than one calculation.

## Dataset

The project includes `data/sample_sales.csv`, containing 20 fictional sales transactions across four product categories:

- Footwear
- Outerwear
- Accessories
- Equipment

Each record includes an order identifier, date, category, product, quantity, unit price and unit cost.

The dataset is synthetic and intended for demonstration and testing.

## Available tools

| Tool | Description |
|---|---|
| `get_total_revenue` | Calculates revenue across all orders |
| `get_total_profit` | Calculates profit across all orders |
| `get_revenue_by_category` | Returns revenue grouped by product category |

Calculations are performed by Python rather than the LLM.

## Installation

This project is part of the [LLM Engineering Lab](https://github.com/efsxanth/llm-engineering-lab) repository.

From the repository root, create and activate a virtual environment if needed.

Install the project in editable mode:

```powershell
python -m pip install -e "./projects/03_data_analysis_agent[dev]"
```

Create a root-level `.env` file using the repository's `.env.example` as a template and configure:

```dotenv
GROQ_API_KEY=your_api_key
GROQ_MODEL=openai/gpt-oss-20b
```

Never commit real API keys.

## Run the application

From the repository root:

```powershell
python .\projects\03_data_analysis_agent\sales_agent.py
```

The demonstration asks:

> Which product category generated the highest revenue?

For the included dataset, the expected answer is Footwear, with £1,210.00 in revenue.

## Automated tests

The project includes tests for data validation, analytical calculations and agent execution.

Offline agent tests use predefined fake LLM responses, allowing tool execution to be tested without making API calls.

Run:

```powershell
python -m pytest .\projects\03_data_analysis_agent\tests -v
```

**Results:** 11 automated tests passed.

## Agent evaluation

The project includes five evaluation questions with expected tool calls and answer values.

Run:

```powershell
python .\projects\03_data_analysis_agent\evaluate_agent.py
```

The evaluation uses live GroqCloud calls and checks:

- Whether the agent selects exactly the expected analytical tools.
- Whether the final answer contains the expected terms or values.

### Recorded results

| Metric | Result |
|---|---|
| Evaluation cases completed | 5/5 |
| Exact expected tool-call matches | 5/5 |
| Answer term checks passed | 5/5 |

These results were obtained on a five-question benchmark using 20 fictional sales records. Answer checks use simple text matching, not comprehensive factual verification. They do not establish general agent accuracy or reliability.

## Code quality

Run Ruff linting and formatting checks:

```powershell
python -m ruff check .\projects\03_data_analysis_agent\
python -m ruff format --check .\projects\03_data_analysis_agent\
```

## Project structure

```text
03_data_analysis_agent/
├── data/
│   └── sample_sales.csv
├── evaluation/
│   └── agent_cases.json
├── tests/
│   ├── test_analytics.py
│   ├── test_data_loader.py
│   └── test_sales_agent.py
├── agent_tools.py
├── analytics.py
├── data_loader.py
├── evaluate_agent.py
├── pyproject.toml
├── README.md
└── sales_agent.py
```

## Limitations

- The application analyses one bundled synthetic CSV dataset.
- It supports a predefined set of read-only analytical tools.
- The example script uses a fixed demonstration question rather than an interactive interface.
- CSV data is loaded from disk when a tool executes.
- The evaluation dataset is small, and the answer checks do not detect every possible error.

## Purpose

This project demonstrates practical AI-agent engineering: tool calling, controlled execution, multi-step workflows, dependency injection, offline testing and basic agent evaluation.