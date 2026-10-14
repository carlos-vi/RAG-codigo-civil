"""Comprueba que todo lo que necesita el proyecto funciona de verdad.

Los endpoints gratuitos de NVIDIA se retiran sin avisar (error 410 "Gone") y
el catalogo de la libreria no es fiable, asi que la unica prueba valida es una
llamada real. Este script hace una peticion minima a cada modelo configurado
en config.py y dice que falla y como arreglarlo.

Uso:
    python src/check_models.py

Termina con codigo 0 si todo esta bien y 1 si algo falla.
"""

from __future__ import annotations

import os
import sys
import time

import requests

import config

NVIDIA_BASE = "https://integrate.api.nvidia.com/v1"
TIMEOUT = 120


def _status(ok: bool, name: str, detail: str) -> bool:
    print(f"[{'OK' if ok else 'FALLO':5}] {name}: {detail}")
    return ok


def _explain(response: requests.Response) -> str:
    if response.status_code == 410:
        return "modelo RETIRADO por NVIDIA (410 Gone): elige otro en src/config.py"
    if response.status_code == 404:
        return "el modelo no existe en tu cuenta (404): revisa el nombre en src/config.py"
    if response.status_code in (401, 403):
        return f"clave rechazada ({response.status_code}): revisa NVIDIA_API_KEY"
    if response.status_code == 429:
        return "limite de peticiones (429): espera un minuto y reintenta"
    return f"HTTP {response.status_code}: {response.text[:120]!r}"


def check_key() -> bool:
    key = os.environ.get("NVIDIA_API_KEY", "")
    if not key or "REEMPLAZA" in key:
        return _status(
            False,
            "NVIDIA_API_KEY",
            "no definida. Copia .env.example a .env y pega tu clave "
            "(gratis en https://build.nvidia.com)",
        )
    return _status(True, "NVIDIA_API_KEY", f"definida ({len(key)} caracteres)")


def _post(path: str, payload: dict) -> requests.Response:
    headers = {"Authorization": f"Bearer {os.environ['NVIDIA_API_KEY']}"}
    return requests.post(f"{NVIDIA_BASE}/{path}", headers=headers, json=payload, timeout=TIMEOUT)


def check_chat(model: str, role: str) -> bool:
    t = time.time()
    try:
        # Los modelos de razonamiento gastan tokens pensando: con un limite
        # bajo devuelven 'content' vacio aunque funcionen.
        r = _post(
            "chat/completions",
            {
                "model": model,
                "temperature": 0,
                "max_tokens": 1500,
                "messages": [{"role": "user", "content": "Responde solo: hola"}],
            },
        )
    except requests.RequestException as exc:
        return _status(False, f"{role} ({model})", f"sin respuesta: {type(exc).__name__}")
    if not r.ok:
        return _status(False, f"{role} ({model})", _explain(r))
    content = r.json()["choices"][0]["message"].get("content") or ""
    return _status(bool(content.strip()), f"{role} ({model})", f"responde en {time.time() - t:.1f}s")


def check_embeddings() -> bool:
    t = time.time()
    try:
        r = _post(
            "embeddings",
            {
                "model": config.EMBED_MODEL,
                "input": ["contrato de compraventa"],
                "input_type": "query",
                "encoding_format": "float",
            },
        )
    except requests.RequestException as exc:
        return _status(False, f"embeddings ({config.EMBED_MODEL})", f"sin respuesta: {type(exc).__name__}")
    if not r.ok:
        return _status(False, f"embeddings ({config.EMBED_MODEL})", _explain(r))
    dims = len(r.json()["data"][0]["embedding"])
    return _status(True, f"embeddings ({config.EMBED_MODEL})", f"{dims} dimensiones, {time.time() - t:.1f}s")


def check_reranker() -> bool:
    if config.RERANK_MODEL is None:
        return _status(True, "reranker", "desactivado (RERANK_MODEL = None)")
    try:
        import reranker

        t = time.time()
        scores = reranker.score("¿qué es la compraventa?", ["La compraventa es un contrato.", "Receta de paella."])
    except ImportError as exc:
        return _status(
            False,
            f"reranker ({config.RERANK_MODEL})",
            f"faltan dependencias ({exc}). Instala: pip install -r requirements-rerank.txt "
            "(o pon RERANK_MODEL = None en config.py para desactivarlo)",
        )
    except Exception as exc:  # descarga del modelo, Smart App Control, etc.
        return _status(False, f"reranker ({config.RERANK_MODEL})", f"{type(exc).__name__}: {str(exc)[:150]}")
    ok = scores[0] > scores[1]
    return _status(ok, f"reranker ({config.RERANK_MODEL})", f"ordena bien, {time.time() - t:.1f}s")


def check_index() -> bool:
    ready = config.PARENTS_PATH.exists() and config.CHILDREN_PATH.exists() and config.CHROMA_DIR.exists()
    return _status(
        ready,
        "indice del Codigo Civil",
        "listo" if ready else "no existe todavia. Ejecuta: python src/ingest.py",
    )


def main() -> int:
    print("Comprobando el proyecto (peticiones minimas a la API de NVIDIA)...\n")
    if not check_key():
        return 1
    results = [check_chat(config.LLM_MODEL, "generacion")]
    if config.JUDGE_MODEL != config.LLM_MODEL:
        results.append(check_chat(config.JUDGE_MODEL, "juez de RAGAS"))
    results += [check_embeddings(), check_reranker(), check_index()]

    print()
    if all(results):
        print("Todo correcto.")
        return 0
    print("Hay problemas: corrige lo marcado como FALLO.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
