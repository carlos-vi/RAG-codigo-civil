"""Latencia del sistema por etapas: reescritura de la consulta, busqueda
hibrida, reranker local y generacion del LLM.

Recorre las 17 preguntas de eval/qa_codigo_civil.jsonl con la configuracion
de src/config.py y guarda el detalle en eval/latency.csv. Los tiempos
dependen del equipo (el reranker corre en CPU) y de la carga de la API de
NVIDIA, asi que varian entre ejecuciones.

Uso (dos o tres llamadas a la API por pregunta):
    python src/bench_latency.py
"""

from __future__ import annotations

import csv
import json
import statistics
import time

from langchain_nvidia_ai_endpoints import ChatNVIDIA

import config
from rag_chain import PROMPT_TEMPLATE, _format_context
from retrieval import retrieve

OUT = config.ROOT / "eval" / "latency.csv"
STAGES = ["reescritura_s", "busqueda_s", "reranker_s", "generacion_s", "total_s"]


def main() -> None:
    config.require_api_key()
    path = config.ROOT / "eval" / "qa_codigo_civil.jsonl"
    questions = [json.loads(l)["question"] for l in path.open(encoding="utf-8") if l.strip()]
    llm = ChatNVIDIA(model=config.LLM_MODEL, temperature=0)

    print("Calentando (carga del indice, BM25 y reranker)...")
    retrieve("¿Qué es la legítima?")

    rows = []
    for q in questions:
        timings: dict = {}
        docs = retrieve(q, timings=timings)
        prompt = PROMPT_TEMPLATE.invoke({"context": _format_context(docs), "question": q})
        for attempt in range(4):
            t = time.perf_counter()
            try:
                llm.invoke(prompt)
                break
            except Exception as exc:  # 429 del tier gratuito: esperar y repetir
                if "429" not in str(exc) or attempt == 3:
                    raise
                time.sleep(15 * (attempt + 1))
        timings["generacion_s"] = time.perf_counter() - t
        timings["total_s"] = sum(timings[s] for s in STAGES[:-1])
        rows.append({"question": q, **{k: round(v, 2) if isinstance(v, float) else v for k, v in timings.items()}})
        print("  ".join(f"{s[:-2]} {timings[s]:4.1f}s" for s in STAGES) + f"  {q[:40]}", flush=True)
        time.sleep(5)  # holgura frente al limite de peticiones

    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n=== Mediana (min-max) sobre {len(rows)} preguntas ===")
    for s in STAGES:
        values = [r[s] for r in rows]
        print(f"{s[:-2]:12} {statistics.median(values):5.1f}s ({min(values):.1f}-{max(values):.1f})")
    print(f"Candidatos al reranker: mediana {statistics.median(r['candidatos'] for r in rows):.0f}")
    print(f"Preguntas reescritas: {sum(r['consultas'] == 2 for r in rows)}/{len(rows)}")
    print(f"\nDetalle en {OUT}")


if __name__ == "__main__":
    main()
