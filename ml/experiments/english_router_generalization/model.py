"""Deterministic dependency-free TF-IDF and linear routing candidates."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

ROUTES = ("VISUAL", "SPEECH", "HYBRID")


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.casefold())


def _terms(text: str, analyzer: str) -> set[str]:
    if analyzer == "word":
        tokens = _words(text)
        return set(tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:])])
    normalized = f" {' '.join(_words(text))} "
    return {normalized[index:index + size] for size in (3, 4, 5)
            for index in range(max(0, len(normalized) - size + 1))}


def fit_vectorizer(texts: list[str], analyzer: str, max_features: int) -> dict:
    if analyzer not in {"word", "char"} or not texts:
        raise ValueError("invalid vectorizer training input")
    document_frequency = Counter(term for text in texts for term in _terms(text, analyzer))
    eligible = [(term, count) for term, count in document_frequency.items() if count >= 2]
    eligible.sort(key=lambda item: (-item[1], item[0]))
    vocabulary = [term for term, _ in eligible[:max_features]]
    idf = [math.log((1 + len(texts)) / (1 + document_frequency[term])) + 1
           for term in vocabulary]
    return {"analyzer": analyzer, "vocabulary": vocabulary, "idf": idf,
            "max_features": max_features, "fit_documents": len(texts)}


def transform(texts: list[str], vectorizer: dict) -> np.ndarray:
    lookup = {term: index for index, term in enumerate(vectorizer["vocabulary"])}
    output = np.zeros((len(texts), len(lookup)), dtype=np.float64)
    idf = np.asarray(vectorizer["idf"], dtype=np.float64)
    for row_index, text in enumerate(texts):
        counts = Counter(term for term in _terms(text, vectorizer["analyzer"])
                         if term in lookup)
        for term, count in counts.items():
            output[row_index, lookup[term]] = (1 + math.log(count)) * idf[lookup[term]]
    norms = np.linalg.norm(output, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return output / norms


def fit_classifier(features: np.ndarray, labels: np.ndarray, regularization: float = 0.05,
                   iterations: int = 450) -> dict:
    if features.ndim != 2 or len(features) != len(labels) or set(labels) != {0, 1, 2}:
        raise ValueError("balanced three-class training data is required")
    x = np.column_stack([np.ones(len(features)), features])
    weights = np.zeros((x.shape[1], 3), dtype=np.float64)
    target = np.eye(3)[labels]
    for step in range(iterations):
        logits = x @ weights
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        gradient = x.T @ (probabilities - target) / len(x)
        gradient[1:] += regularization * weights[1:] / len(x)
        weights -= (0.8 / math.sqrt(step + 1)) * gradient
    return {"weights": weights.tolist(), "regularization": regularization,
            "iterations": iterations, "parameters": int(weights.size)}


def fit_model(rows: list[dict], architecture: str) -> dict:
    texts = [row["query"] for row in rows]
    labels = np.asarray([ROUTES.index(row["route"]) for row in rows])
    analyzers = {"word_tfidf": (("word", 5000),),
                 "char_tfidf": (("char", 8000),),
                 "word_char_tfidf": (("word", 5000), ("char", 8000))}.get(architecture)
    if analyzers is None:
        raise ValueError("unknown router architecture")
    vectorizers = [fit_vectorizer(texts, analyzer, maximum)
                   for analyzer, maximum in analyzers]
    features = np.column_stack([transform(texts, vectorizer)
                                for vectorizer in vectorizers])
    return {"schema_version": "1.0.0", "architecture": architecture,
            "routes": list(ROUTES), "vectorizers": vectorizers,
            "classifier": fit_classifier(features, labels), "fallback_threshold": 0.0}


def predict(model: dict, texts: list[str], fallback_threshold: float | None = None):
    if tuple(model.get("routes", [])) != ROUTES:
        raise ValueError("router artifact route schema mismatch")
    if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("router queries must be non-empty strings")
    features = np.column_stack([transform(texts, vectorizer)
                                for vectorizer in model["vectorizers"]])
    x = np.column_stack([np.ones(len(features)), features])
    logits = x @ np.asarray(model["classifier"]["weights"])
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    indices = probabilities.argmax(axis=1)
    confidence = probabilities[np.arange(len(indices)), indices]
    routes = [ROUTES[index] for index in indices]
    threshold = model.get("fallback_threshold", 0.0) if fallback_threshold is None else fallback_threshold
    routes = ["HYBRID" if score < threshold else route
              for route, score in zip(routes, confidence)]
    return routes, confidence, probabilities


def save_model(model: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, sort_keys=True, separators=(",", ":")), encoding="utf-8")


def load_model(path: Path) -> dict:
    model = json.loads(path.read_text(encoding="utf-8"))
    if model.get("schema_version") != "1.0.0" or tuple(model.get("routes", [])) != ROUTES:
        raise ValueError("unsupported router artifact")
    if not model.get("vectorizers") or not model.get("classifier", {}).get("weights"):
        raise ValueError("incomplete router artifact")
    return model


def artifact_size(model: dict) -> int:
    return len(json.dumps(model, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def model_memory_bytes(model: dict) -> int:
    weights = np.asarray(model["classifier"]["weights"])
    idf = sum(np.asarray(vectorizer["idf"]).nbytes for vectorizer in model["vectorizers"])
    vocabulary = sum(sum(len(term.encode("utf-8")) for term in vectorizer["vocabulary"])
                     for vectorizer in model["vectorizers"])
    return int(weights.nbytes + idf + vocabulary)
