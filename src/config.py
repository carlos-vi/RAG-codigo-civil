"""Configuracion central: modelos de NVIDIA, rutas y parametros del RAG sobre el Codigo Civil.

La clave de la API se lee de la variable de entorno NVIDIA_API_KEY (o de un
archivo .env en la raiz, que esta en .gitignore). Se obtiene gratis en
https://build.nvidia.com
"""

from __future__ import annotations

import os
import pathlib

from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data" / "codigo_civil"
XML_PATH = DATA_DIR / "codigo_civil.xml"
# Legislacion consolidada del BOE (datos abiertos): texto vigente con la
# estructura del Codigo (libros, titulos, capitulos, articulos).
BOE_ID = "BOE-A-1889-4763"
XML_URL = f"https://www.boe.es/datosabiertos/api/legislacion-consolidada/id/{BOE_ID}/texto"

CHROMA_DIR = ROOT / "chroma_db" / "codigo_civil"
COLLECTION_NAME = "codigo_civil"
PARENTS_PATH = DATA_DIR / "parents.json"
CHILDREN_PATH = DATA_DIR / "children.json"

# Modelos servidos por la API de NVIDIA (build.nvidia.com). Los endpoints
# gratuitos se retiran con frecuencia (error 410 "Gone"): verificar con una
# llamada real, no solo con el catalogo (ver README, seccion 5; python src/check_models.py).
LLM_MODEL = "nvidia/nemotron-3-super-120b-a12b"
# Juez de RAGAS: el mismo modelo que genera. Lo ideal seria otra familia
# para evitar sesgo de autoevaluacion, pero un juez de razonamiento lento
# (z-ai/glm-5.3) hacia la evaluacion inviable en el tier gratuito.
JUDGE_MODEL = LLM_MODEL
EMBED_MODEL = "nvidia/nemotron-3-embed-1b"
# Reranker LOCAL (cross-encoder multilingue de Hugging Face, ~470 MB, se
# descarga la primera vez). Los rerankers alojados de NVIDIA estan retirados.
# None desactiva la etapa. Requiere requirements-rerank.txt.
RERANK_MODEL: str | None = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# Reescritura de la consulta: un LLM traduce la pregunta a terminologia
# juridica antes de buscar ("plazo para reclamar una deuda" -> "prescripcion
# de las acciones personales"). Ayuda con las preguntas en lenguaje
# cotidiano a cambio de una llamada mas al LLM por pregunta. False la desactiva.
REWRITE_QUERY = True

# Chunking jerarquico: el PADRE es el articulo completo; los HIJOS son
# fragmentos de los articulos largos (los cortos son un unico hijo). Se busca
# con hijos y se entrega al LLM el articulo entero.
CHILD_CHUNK_SIZE = 700
CHILD_CHUNK_OVERLAP = 100

# Limite del tier gratuito: ~40 peticiones/minuto.
EMBED_BATCH_SIZE = 32
EMBED_PAUSE_SECONDS = 2.0


def require_api_key() -> str:
    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        raise SystemExit(
            "Falta NVIDIA_API_KEY. Crea una clave gratis en https://build.nvidia.com "
            "y ponla en un archivo .env (ver .env.example) o como variable de entorno."
        )
    return key
