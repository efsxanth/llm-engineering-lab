from dataclasses import dataclass

from langchain_core.documents import Document


@dataclass
class QAResult:
    """A generated answer and its supporting documents."""

    answer: str
    sources: list[Document]
