"""Evalúa el pipeline RAG (Código Civil) con RAGAS.

Ejecuta cada pregunta del dataset de referencia (eval/qa_codigo_civil.jsonl) a
través del pipeline (retrieve -> generate), recopila pregunta, respuesta
generada, contextos recuperados y respuesta de referencia, y calcula las
métricas de RAGAS. El juez es el mismo LLM que genera (config.JUDGE_MODEL),
lo que puede inflar las metricas: ver README, seccion 4.

Uso:
    python src/evaluate.py
    python src/evaluate.py --limit 1   # prueba rapida con menos preguntas
"""

from __future__ import annotations

import argparse
import json
import pathlib

from datasets import Dataset
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig

import config
from rag_chain import answer_question

ROOT = pathlib.Path(__file__).resolve().parent.parent
QA_DATASET_PATH = ROOT / "eval" / "qa_codigo_civil.jsonl"
RESULTS_CSV_PATH = ROOT / "eval" / "results_codigo_civil.csv"


def load_qa_dataset() -> list[dict]:
    records = []
    with QA_DATASET_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_pipeline_over_dataset(records: list[dict]) -> Dataset:
    questions, answers, contexts, ground_truths = [], [], [], []

    for record in records:
        question = record["question"]
        print(f"Procesando: {question}")

        answer, sources = answer_question(question)

        questions.append(question)
        answers.append(answer)
        contexts.append([doc.page_content for doc in sources])
        ground_truths.append(record["ground_truth"])

    return Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalua el pipeline RAG con RAGAS.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Usar solo las primeras N preguntas del dataset (util para pruebas rapidas).",
    )
    args = parser.parse_args()

    records = load_qa_dataset()
    if args.limit is not None:
        records = records[: args.limit]
    print(f"Cargadas {len(records)} preguntas de evaluación.")

    dataset = run_pipeline_over_dataset(records)

    # El juez es un modelo de razonamiento: si el limite de tokens es bajo gasta
    # todo pensando y RAGAS falla con LLMDidNotFinishException (metricas nan).
    judge_llm = ChatNVIDIA(
        model=config.JUDGE_MODEL, temperature=0, max_completion_tokens=8192
    )
    judge_embeddings = NVIDIAEmbeddings(model=config.EMBED_MODEL, truncate="END")

    # Tier gratuito de NVIDIA: ~40 peticiones/minuto. RAGAS hace varias
    # llamadas por pregunta y metrica, asi que se limita la concurrencia.
    run_config = RunConfig(timeout=600, max_workers=2, max_retries=10)

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=run_config,
    )

    df = result.to_pandas()
    df.to_csv(RESULTS_CSV_PATH, index=False)

    print("\n=== Métricas agregadas ===")
    for metric_name in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        if metric_name in df.columns:
            print(f"{metric_name}: {df[metric_name].mean():.3f}")

    print(f"\nResultados detallados guardados en {RESULTS_CSV_PATH}")


if __name__ == "__main__":
    main()
