"""Reranker local: cross-encoder multilingue ejecutado con transformers + torch.

Lee pregunta y articulo juntos y puntua su relevancia, mucho mas preciso que
la similitud de embeddings. Requiere las dependencias de
requirements-rerank.txt (torch 2.2.2 con versiones antiguas fijadas por
Smart App Control). Si no estan instaladas, retrieval.py sigue funcionando
sin reranking.
"""

from __future__ import annotations

from functools import lru_cache

import config

BATCH_SIZE = 8
MAX_LENGTH = 384  # tokens de pregunta + articulo; los articulos largos se truncan


@lru_cache(maxsize=1)
def _load():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(max(1, torch.get_num_threads()))
    tokenizer = AutoTokenizer.from_pretrained(config.RERANK_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(config.RERANK_MODEL).eval()
    return torch, tokenizer, model


def score(question: str, texts: list[str]) -> list[float]:
    """Puntuacion de relevancia de cada texto respecto a la pregunta."""
    torch, tokenizer, model = _load()
    scores: list[float] = []
    with torch.no_grad():
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start : start + BATCH_SIZE]
            enc = tokenizer(
                [question] * len(batch),
                batch,
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
                return_tensors="pt",
            )
            scores += model(**enc).logits.squeeze(-1).tolist()
    return scores
