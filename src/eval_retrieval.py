"""Evaluacion de la recuperacion con preguntas en lenguaje cotidiano.

Complementa a evaluate.py (RAGAS): no usa LLM juez, solo comprueba si el
articulo que responde a la pregunta llega a los `top_k` que recibe el LLM y
en que posicion. Es determinista salvo por la API de embeddings.

Dataset: eval/qa_cotidiano.jsonl. Cada pregunta lista los articulos que la
responden; cuenta como acierto que CUALQUIERA de ellos este en el top_k. Las
preguntas estan divididas en:
- dev:  para ajustar cambios (p. ej. la reescritura de la consulta);
- test: para medir el resultado final, sin ajustar nada con ellas;
- cita: preguntas que citan el articulo por numero (prueba de regresion).

Uso:
    python src/eval_retrieval.py                  # todas las preguntas
    python src/eval_retrieval.py --split dev
    python src/eval_retrieval.py --out eval/retrieval_cotidiano.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import time

import config
from retrieval import retrieve

DATASET = config.ROOT / "eval" / "qa_cotidiano.jsonl"
TOP_K = 6


def load(split: str | None) -> list[dict]:
    rows = [json.loads(l) for l in DATASET.open(encoding="utf-8") if l.strip()]
    return [r for r in rows if split is None or r["split"] == split]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalua la recuperacion (acierto en top_k).")
    parser.add_argument("--split", choices=["dev", "test", "cita"], default=None)
    parser.add_argument("--out", default=None, help="CSV con el detalle por pregunta")
    args = parser.parse_args()

    config.require_api_key()
    rows = []
    for record in load(args.split):
        docs = retrieve(record["question"], top_k=TOP_K)
        got = [d.metadata["articulo"] for d in docs]
        positions = [got.index(a) + 1 for a in record["articulos"] if a in got]
        rank = min(positions) if positions else None
        rows.append(
            {
                "split": record["split"],
                "question": record["question"],
                "esperados": " ".join(record["articulos"]),
                "posicion": rank or "",
                "recuperados": " | ".join(got),
            }
        )
        print(f"[{record['split']:4}] {str(rank or '-'):>2}  {record['question'][:60]:60}  {got[:4]}", flush=True)
        time.sleep(1)  # holgura frente al limite de peticiones

    print()
    for split in sorted({r["split"] for r in rows}):
        sel = [r for r in rows if r["split"] == split]
        hits = [r for r in sel if r["posicion"] != ""]
        top1 = [r for r in hits if r["posicion"] == 1]
        mrr = sum(1 / r["posicion"] for r in hits) / len(sel)
        print(f"{split:4}: acierto@{TOP_K} {len(hits)}/{len(sel)}  "
              f"en 1.a posicion {len(top1)}/{len(sel)}  MRR {mrr:.2f}")

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nDetalle en {args.out}")


if __name__ == "__main__":
    main()
