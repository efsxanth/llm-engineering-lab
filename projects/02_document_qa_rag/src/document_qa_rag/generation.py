from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = """
You are a helpful product recommendation assistant.

Answer the user's question using only the supplied product information.

Rules:
- Recommend only products supported by the context.
- Explain recommendations using explicitly stated product features.
- Do not invent specifications, performance claims, or suitability
  for activities that the descriptions do not establish.
- Clearly distinguish factual product details from recommendations.
- Ignore products that are irrelevant to the question.
- If the context does not contain enough information, say so.
- Keep the answer concise and specific.

Context:
{context}

Question:
{question}
"""


rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)


def generate_answer(
    question: str,
    documents: list[Document],
    chat: BaseChatModel,
) -> str:
    """Generate an answer using retrieved documents."""

    if not isinstance(question, str):
        raise TypeError("Question must be a string.")

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if not documents:
        return "I cannot answer without supporting documents."

    context = "\n\n".join(document.page_content for document in documents)

    chain = rag_prompt | chat | StrOutputParser()

    return chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )
