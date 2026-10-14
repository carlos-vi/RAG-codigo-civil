"""Descarga el Codigo Civil espanol (BOE, legislacion consolidada), lo divide
por articulos y lo indexa en Chroma con embeddings de NVIDIA.

- PADRE = articulo completo (con su jerarquia: libro > titulo > capitulo > seccion).
- HIJOS = fragmentos de ~700 caracteres del articulo; los articulos cortos
  son un unico hijo. Se buscan los hijos (precisos) y se entrega el padre.

Uso:
    python src/ingest.py

Es reanudable contando lo ya indexado: si cambia el parser o el XML, borrar
chroma_db/codigo_civil/ antes de relanzar (ver README, seccion 2).
"""

from __future__ import annotations

import json
import re
import time
import xml.etree.ElementTree as ET

import requests
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

# Nivel jerarquico segun el rotulo del encabezado en el XML del BOE.
LEVELS = [
    # Los grupos de disposiciones (DISPOSICIONES TRANSITORIAS, ADICIONALES,
    # DISPOSICION FINAL...) van en mayusculas; "Disposiciones generales" (una
    # seccion normal) no debe confundirse con ellos.
    ("libro", re.compile(r"(?i:LIBRO)\b|DISPOSICI[ÓO]N(?:ES)?\b")),
    ("titulo", re.compile(r"^T[ÍI]TULO\b", re.IGNORECASE)),
    ("capitulo", re.compile(r"^CAP[ÍI]TULO\b", re.IGNORECASE)),
    ("seccion", re.compile(r"^(Secci[óo]n|Subsecci[óo]n)\b", re.IGNORECASE)),
]
# Admite sufijos: "Articulo 94 bis.", "Articulo 172 ter."
ARTICLE_RE = re.compile(r"^Art[íi]culo\s+(.+?)\.?\s*$", re.IGNORECASE)


def download_xml() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    if config.XML_PATH.exists():
        print(f"XML ya descargado en {config.XML_PATH}")
        return
    print(f"Descargando {config.XML_URL} ...")
    response = requests.get(
        config.XML_URL, headers={"Accept": "application/xml"}, timeout=120
    )
    response.raise_for_status()
    config.XML_PATH.write_bytes(response.content)
    print(f"Guardado en {config.XML_PATH} ({len(response.content) // 1024} KB)")


def _text(p: ET.Element) -> str:
    return re.sub(r"\s+", " ", "".join(p.itertext())).strip()


def _heading(block: ET.Element) -> tuple[str, str] | None:
    """Devuelve (nivel, rotulo completo) de un bloque 'encabezado'."""
    # Solo la version vigente: cada encabezado conserva sus rotulos historicos
    # y leerlos todos duplicaba la jerarquia.
    versions = block.findall("version")
    source = versions[-1] if versions else block
    paragraphs = [
        _text(p) for p in source.iter("p") if p.get("class") and not p.get("class").startswith("nota")
    ]
    paragraphs = [t for t in paragraphs if t]
    if not paragraphs:
        return None
    for level, regex in LEVELS:
        if regex.match(paragraphs[0]):
            return level, " - ".join(paragraphs)
    return None


def parse_articles(xml_path=None) -> list[dict]:
    """Un dict por articulo vigente: {articulo, etiqueta, jerarquia, text, ...}.

    `xml_path` permite parsear otro XML (los tests usan uno pequeno).
    """
    root = ET.parse(xml_path or config.XML_PATH).getroot()
    context = {name: "" for name, _ in LEVELS}
    order = [name for name, _ in LEVELS]
    articles: list[dict] = []

    for block in root.findall(".//texto/bloque"):
        kind = block.get("tipo")
        if kind == "encabezado":
            parsed = _heading(block)
            if parsed:
                level, label = parsed
                context[level] = label
                # Al abrir un nivel se reinician los inferiores.
                for lower in order[order.index(level) + 1 :]:
                    context[lower] = ""
        elif kind == "precepto":
            versions = block.findall("version")
            if not versions:
                continue
            current = versions[-1]  # ultima version = texto vigente
            paragraphs = [
                (p.get("class"), _text(p))
                for p in current.iter("p")
                if p.get("class") and not p.get("class").startswith("nota")
            ]
            heading = next((t for c, t in paragraphs if c == "articulo"), "")
            # Algunas secciones llegan como "precepto" sin numero de articulo:
            # son rotulos de seccion, no articulos.
            if not heading:
                section = next((t for c, t in paragraphs if c == "seccion"), "")
                if section:
                    context["seccion"] = section
                continue
            body = [t for c, t in paragraphs if c != "articulo" and t]
            text = "\n".join(body)
            # Articulos derogados o sin contenido: ruido para QA.
            if len(text) < 30 or text.lower().startswith("(derogado"):
                continue
            match = ARTICLE_RE.match(heading)
            if match:
                number = match.group(1)
                label = f"Artículo {number}"
            else:
                # Disposiciones ("Primera.", "Segunda."...): se anteponen el
                # grupo al que pertenecen (adicionales, transitorias...).
                group = context["libro"].split(" - ")[0].capitalize()
                number = f"{group} {heading.rstrip('.')}".strip()
                label = number
            articles.append(
                {
                    "articulo": number,
                    "etiqueta": label,
                    "libro": context["libro"],
                    "titulo": context["titulo"],
                    "capitulo": context["capitulo"],
                    "seccion": context["seccion"],
                    "text": text,
                }
            )
    return articles


def _hierarchy(a: dict) -> str:
    return " > ".join(x for x in (a["libro"], a["titulo"], a["capitulo"], a["seccion"]) if x)


def build_chunks(articles: list[dict]) -> tuple[list[dict], list[Document]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHILD_CHUNK_SIZE,
        chunk_overlap=config.CHILD_CHUNK_OVERLAP,
        separators=["\n", ". ", " ", ""],
    )
    parents: list[dict] = []
    children: list[Document] = []
    for parent_id, a in enumerate(articles):
        header = f"{a['etiqueta']}."
        parents.append(
            {
                "parent_id": parent_id,
                "articulo": a["articulo"],
                "etiqueta": a["etiqueta"],
                "jerarquia": _hierarchy(a),
                "text": f"{header}\n{a['text']}",
            }
        )
        for piece in splitter.split_text(a["text"]):
            children.append(
                Document(
                    # El encabezado en cada hijo ayuda a recuperar por numero
                    # de articulo y situa el fragmento.
                    page_content=f"{header} {piece}",
                    metadata={
                        "parent_id": parent_id,
                        "articulo": a["articulo"],
                        "chunk_id": len(children),
                    },
                )
            )
    return parents, children


def build_vectorstore(children: list[Document]) -> Chroma:
    embeddings = NVIDIAEmbeddings(model=config.EMBED_MODEL, truncate="END")
    store = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(config.CHROMA_DIR),
    )
    # Reanudable: salta lo ya indexado si se relanza tras un corte/429.
    done = store._collection.count()
    batch = config.EMBED_BATCH_SIZE
    for start in range(done, len(children), batch):
        chunk = children[start : start + batch]
        for attempt in range(5):
            try:
                store.add_documents(chunk, ids=[str(d.metadata["chunk_id"]) for d in chunk])
                break
            except Exception as exc:  # 429 / errores transitorios de red
                wait = 10 * 2**attempt
                print(f"  error ({exc.__class__.__name__}), reintento en {wait}s")
                time.sleep(wait)
        else:
            raise SystemExit(f"Fallo persistente en el lote que empieza en {start}")
        print(f"  indexados {min(start + batch, len(children))}/{len(children)}")
        time.sleep(config.EMBED_PAUSE_SECONDS)
    return store


def main() -> None:
    config.require_api_key()
    download_xml()

    articles = parse_articles()
    parents, children = build_chunks(articles)
    print(f"{len(parents)} articulos (padres) y {len(children)} fragmentos (hijos).")

    # Se persisten para que el retrieval (BM25 y expansion a padres) no tenga
    # que volver a procesar el XML en cada pregunta.
    config.PARENTS_PATH.write_text(json.dumps(parents, ensure_ascii=False), encoding="utf-8")
    config.CHILDREN_PATH.write_text(
        json.dumps(
            [{"text": c.page_content, **c.metadata} for c in children], ensure_ascii=False
        ),
        encoding="utf-8",
    )

    build_vectorstore(children)
    print(f"Vector store en {config.CHROMA_DIR} (coleccion '{config.COLLECTION_NAME}').")


if __name__ == "__main__":
    main()
