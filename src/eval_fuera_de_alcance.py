"""Comprueba que el sistema declina responder preguntas que el Codigo Civil no
contesta (otras leyes, temas sin relacion, articulos inexistentes o
derogados) en vez de inventar.

Guarda pregunta, articulos recuperados y respuesta en
eval/fuera_de_alcance.json para revisarlos a mano: no hay metrica
automatica, la valoracion esta en el README.

Uso (9 preguntas, ~20 s cada una):
    python src/eval_fuera_de_alcance.py
"""

from __future__ import annotations

import json
import time

import config
from rag_chain import answer_question

OUT = config.ROOT / "eval" / "fuera_de_alcance.json"

QUESTIONS = [
    ("otra ley: fiscal", "¿Cuál es el tipo general del IVA en España?"),
    ("otra ley: tráfico", "¿Cuál es la multa por superar el límite de velocidad en autopista?"),
    ("otra ley: penal", "¿Qué pena tiene el robo con violencia?"),
    ("otra ley: laboral", "¿Cuántos días de indemnización por año trabajado corresponden en un despido improcedente?"),
    ("sin relación", "¿Cuál es la capital de Francia?"),
    ("sin relación", "Dame una receta de paella valenciana."),
    ("art. inexistente", "¿Qué dice el artículo 5000 del Código Civil?"),
    ("art. derogado", "¿Qué dice el artículo 315 del Código Civil sobre la mayoría de edad?"),
    ("tocada de refilón", "¿Cuánto tiempo tengo para reclamar a mi casero la devolución de la fianza?"),
]


def main() -> None:
    results = []
    for kind, question in QUESTIONS:
        answer, docs = answer_question(question)
        results.append(
            {
                "tipo": kind,
                "pregunta": question,
                "recuperado": [d.metadata["etiqueta"] for d in docs],
                "respuesta": answer,
            }
        )
        print(f"\n### [{kind}] {question}\n{answer}", flush=True)
        time.sleep(2)  # holgura frente al limite de peticiones

    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"\nGuardado en {OUT}")


if __name__ == "__main__":
    main()
