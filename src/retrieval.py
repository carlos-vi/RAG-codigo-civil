"""Retrieval sobre el Codigo Civil:

0. Los articulos citados por numero ("articulo 1") se recuperan directamente
   y van primero. Si config.REWRITE_QUERY, un LLM reescribe la pregunta en
   terminos juridicos y las etapas 1-2 se hacen con ambas consultas.
1. Hibrido (BM25 + densa) sobre chunks HIJOS pequenos: 20 por rama,
   fusionados con RRF ponderado.
2. Los hijos se expanden a su PADRE (el articulo completo), sin duplicados
   (~33 articulos candidatos; con reescritura, los 10 primeros de cada
   consulta).
3. Reranker local (cross-encoder, src/reranker.py) puntua el articulo entero
   y reordena; opcional (config.RERANK_MODEL = None lo desactiva).

BM25 aporta coincidencia lexica y la busqueda densa similitud semantica; el
reranker corrige el orden con una lectura conjunta de pregunta y pasaje.
BM25 tokeniza con `tokenize`: minusculas, sin tildes y sin signos de
puntuacion (se conservan letras y numeros), para que "808" coincida con
"808." y "legitima?" con "legítima".
"""

from __future__ import annotations

import json
import re
import time
import warnings
from functools import lru_cache

from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

import config

BM25_WEIGHT = 0.4
DENSE_WEIGHT = 0.6
CANDIDATES = 20  # por rama (BM25 / densa) antes de fusionar
# Quita tildes y dieresis pero conserva la ñ ("año" no debe ser "ano").
_ACCENTS = str.maketrans("áéíóúàèìòùü", "aeiouaeiouu")


def tokenize(text: str) -> list[str]:
    """Tokens para BM25: minusculas, sin tildes y sin puntuacion. El
    tokenizador por defecto (text.split()) dejaba "808." o "legítima?" pegados
    a la puntuacion y no coincidian con la pregunta."""
    return re.findall(r"\w+", text.lower().translate(_ACCENTS))


# "articulo 1", "art. 1902", "art 94 bis", "articulos 1 y 2" (sobre texto en
# minusculas y sin tildes).
_ARTICLE_REF = re.compile(r"\bart(?:iculos?)?\.?\s*(\d+(?:\s+(?:bis|ter|quater))?)\b")
_ARTICLE_LIST = re.compile(r"^(?:\s*(?:,|y|e)\s*(\d+(?:\s+(?:bis|ter|quater))?)\b)+")


def cited_articles(question: str) -> list[str]:
    """Numeros de articulo que cita la pregunta, en orden y sin repetir.

    Buscar un articulo por su numero no funciona bien con busqueda semantica:
    el "1" aparece en cientos de fragmentos y el reranker no sabe que articulo
    es "el 1". Por eso los articulos citados se recuperan directamente."""
    text = question.lower().translate(_ACCENTS)
    found: list[str] = []
    for match in _ARTICLE_REF.finditer(text):
        numbers = [match.group(1)]
        rest = _ARTICLE_LIST.match(text[match.end() :])
        if rest:
            numbers += re.findall(r"\d+(?:\s+(?:bis|ter|quater))?", rest.group(0))
        for n in numbers:
            n = re.sub(r"\s+", " ", n)
            if n not in found:
                found.append(n)
    return found


# Sin ejemplos de instituciones juridicas a proposito: el LLM tiende a
# reciclarlos en preguntas que no tienen que ver, y sesgarian la evaluacion.
REWRITE_PROMPT = """Reformula la siguiente pregunta de un ciudadano como una consulta de búsqueda sobre el Código Civil español, usando la terminología jurídica que emplea el Código y nombrando las instituciones jurídicas implicadas. No incluyas números de artículo. Si la pregunta no trata de derecho civil, responde exactamente: NO_APLICA. Devuelve solo la consulta, en una línea.

Pregunta: {question}"""


REWRITE_TIMEOUT_SECONDS = 60
# Con reescritura, candidatos de cada consulta que pasan al reranker: asi su
# coste es parecido al de una sola busqueda.
REWRITE_CANDIDATES_PER_QUERY = 10


@lru_cache(maxsize=256)
def rewrite_query(question: str) -> str:
    """Traduce una pregunta en lenguaje cotidiano a terminos juridicos.

    "plazo para reclamar una deuda" no recupera el art. 1964 porque el Codigo
    habla de "prescripcion de las acciones personales". Sin numeros de
    articulo: el LLM podria inventarlos y BM25 los premiaria.

    Peticion directa con tiempo limite: una llamada a la API que no respondia
    llego a colgar el proceso. Si falla o tarda, se busca con la pregunta
    original."""
    import requests

    try:
        # El tier gratuito limita las peticiones por minuto (429): se
        # reintenta un par de veces antes de renunciar a la reescritura.
        for attempt in range(3):
            response = requests.post(
                "https://integrate.api.nvidia.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {config.require_api_key()}"},
                json={
                    "model": config.LLM_MODEL,
                    "temperature": 0,
                    "messages": [{"role": "user", "content": REWRITE_PROMPT.format(question=question)}],
                },
                timeout=REWRITE_TIMEOUT_SECONDS,
            )
            if response.status_code != 429 or attempt == 2:
                break
            time.sleep(5 * (attempt + 1))
        response.raise_for_status()
        text = response.json()["choices"][0]["message"].get("content") or ""
    except (requests.RequestException, KeyError, ValueError) as exc:
        warnings.warn(f"Reescritura de la consulta fallida ({type(exc).__name__}); se usa la pregunta original.")
        return question
    text = " ".join(text.split())
    # Fuera del derecho civil no se reescribe: se busca solo con la pregunta.
    return question if not text or "NO_APLICA" in text else text


@lru_cache(maxsize=1)
def _parents() -> dict[int, dict]:
    items = json.loads(config.PARENTS_PATH.read_text(encoding="utf-8"))
    return {p["parent_id"]: p for p in items}


@lru_cache(maxsize=1)
def _parents_by_number() -> dict[str, dict]:
    return {p["articulo"]: p for p in _parents().values()}


@lru_cache(maxsize=1)
def _hybrid_retriever() -> EnsembleRetriever:
    rows = json.loads(config.CHILDREN_PATH.read_text(encoding="utf-8"))
    docs = [
        Document(page_content=r["text"], metadata={k: v for k, v in r.items() if k != "text"})
        for r in rows
    ]
    bm25 = BM25Retriever.from_documents(docs, preprocess_func=tokenize)
    bm25.k = CANDIDATES

    store = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=NVIDIAEmbeddings(model=config.EMBED_MODEL, truncate="END"),
        persist_directory=str(config.CHROMA_DIR),
    )
    dense = store.as_retriever(search_kwargs={"k": CANDIDATES})
    return EnsembleRetriever(retrievers=[bm25, dense], weights=[BM25_WEIGHT, DENSE_WEIGHT])


def _rerank(queries: list[str], parents: list[dict]) -> list[dict]:
    """Reordena los articulos candidatos con el cross-encoder local. Con varias
    consultas (original y reescrita) cada articulo se queda con su mejor
    puntuacion. Si no hay reranker configurado o faltan sus dependencias,
    conserva el orden de entrada (fusion hibrida)."""
    if config.RERANK_MODEL is None:
        return parents
    try:
        import reranker

        texts = [p["text"] for p in parents]
        scores = [max(s) for s in zip(*(reranker.score(q, texts) for q in queries))]
    except ImportError as exc:
        warnings.warn(
            f"Reranker desactivado ({exc}). Instala requirements-rerank.txt para activarlo."
        )
        return parents
    order = sorted(range(len(parents)), key=lambda i: scores[i], reverse=True)
    return [parents[i] for i in order]


def _hybrid_parents(query: str) -> list[dict]:
    """Busqueda hibrida sobre los hijos -> padres unicos en el orden de la fusion."""
    all_parents = _parents()
    out, seen = [], set()
    for child in _hybrid_retriever().invoke(query):
        pid = child.metadata["parent_id"]
        if pid not in seen:
            seen.add(pid)
            out.append(all_parents[pid])
    return out


def retrieve(question: str, top_k: int = 6, timings: dict | None = None) -> list[Document]:
    """Devuelve hasta `top_k` documentos PADRE (articulos) por relevancia. Los
    articulos citados por numero en la pregunta van primero; el resto del
    contexto sale de la busqueda hibrida (con la pregunta y, si
    config.REWRITE_QUERY, su version en terminos juridicos) y el reranker.

    Si se pasa `timings`, anota en el los segundos de cada etapa
    (reescritura, busqueda, reranker); lo usa src/bench_latency.py."""
    clock = time.perf_counter
    t = clock()
    by_number = _parents_by_number()
    # Un numero que no existe (o derogado) simplemente no se encuentra.
    cited = [by_number[n] for n in cited_articles(question) if n in by_number][:top_k]
    cited_ids = {p["parent_id"] for p in cited}

    # Lenguaje cotidiano -> terminos juridicos. Se buscan candidatos con
    # ambas consultas y solo los primeros de cada una pasan al reranker, para
    # que su coste sea parecido al de una sola busqueda. Si la reescritura
    # falla (devuelve la pregunta), se sigue como sin reescritura.
    rewritten = rewrite_query(question) if config.REWRITE_QUERY else question
    t_rewrite, t = clock() - t, clock()
    if rewritten != question:
        queries, per_query = [question, rewritten], REWRITE_CANDIDATES_PER_QUERY
    else:
        queries, per_query = [question], None

    unique, seen = [], set(cited_ids)
    for query in queries:
        for p in _hybrid_parents(query)[:per_query]:
            if p["parent_id"] not in seen:
                seen.add(p["parent_id"])
                unique.append(p)
    t_search, t = clock() - t, clock()

    ranked = cited + _rerank(queries, unique)
    if timings is not None:
        timings.update(
            reescritura_s=t_rewrite, busqueda_s=t_search, reranker_s=clock() - t,
            consultas=len(queries), candidatos=len(unique),
        )
    return [
        Document(
            page_content=p["text"],
            metadata={
                "parent_id": p["parent_id"],
                "articulo": p["articulo"],
                "etiqueta": p["etiqueta"],
                "jerarquia": p["jerarquia"],
                "source": "Código Civil (BOE-A-1889-4763)",
            },
        )
        for p in ranked[:top_k]
    ]


if __name__ == "__main__":
    for i, doc in enumerate(retrieve("¿Qué es la legítima?"), 1):
        preview = doc.page_content[:120].replace("\n", " ")
        print(f"[{i}] {doc.metadata['etiqueta']}: {preview}...")
