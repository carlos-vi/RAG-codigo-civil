"""Cadena RAG completa: retrieval (citas, reescritura, hibrido, reranker) -> generacion
con un LLM de la API de NVIDIA, citando los articulos usados.

Uso:
    python src/rag_chain.py --question "¿A qué edad se alcanza la mayoría de edad?"
"""

from __future__ import annotations

import argparse

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA

import config
from retrieval import retrieve

PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    """Eres un asistente jurídico que responde preguntas sobre el Código Civil español basándote ÚNICAMENTE en los artículos proporcionados en el contexto. Si la respuesta no está en ellos, dilo explícitamente en vez de inventarla. No des asesoramiento legal personalizado: explica lo que dice la norma.

Cita siempre los artículos en los que te apoyas, con el formato "art. N" (por ejemplo, art. 1902), y la numeración [1], [2] del contexto. Responde en el mismo idioma que la pregunta.

Contexto:
{context}

Pregunta: {question}

Respuesta:"""
)


def _format_context(documents: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(documents, 1):
        m = doc.metadata
        parts.append(f"[{i}] ({m['jerarquia']})\n{doc.page_content}")
    return "\n\n".join(parts)


def answer_question(
    question: str,
    top_k: int = 6,
    model: str = config.LLM_MODEL,
) -> tuple[str, list[Document]]:
    config.require_api_key()
    docs = retrieve(question, top_k=top_k)

    llm = ChatNVIDIA(model=model, temperature=0)
    response = (PROMPT_TEMPLATE | llm).invoke(
        {"context": _format_context(docs), "question": question}
    )
    return response.content, docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Pregunta al RAG sobre el Codigo Civil.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--model", default=config.LLM_MODEL, help="Modelo de build.nvidia.com")
    args = parser.parse_args()

    answer, sources = answer_question(args.question, model=args.model)

    print("\n=== Respuesta ===")
    print(answer)

    print("\n=== Fuentes usadas como contexto ===")
    for i, doc in enumerate(sources, 1):
        print(f"[{i}] {doc.metadata['etiqueta']}  ({doc.metadata['jerarquia']})")


if __name__ == "__main__":
    main()
