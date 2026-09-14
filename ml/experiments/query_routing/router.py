"""Cheap deterministic and learned query-text routers."""

import math
import re

import numpy as np

ROUTES = ("VISUAL", "SPEECH", "HYBRID")
SPEECH_WORDS = {"say", "says", "said", "mention", "mentions", "mentioned",
                "explain", "explains", "explained", "discuss", "discusses",
                "talk", "talks", "describe", "describes", "hear", "spoken"}
VISUAL_WORDS = {"appear", "appears", "visible", "screen", "shown", "showing",
                "slide", "diagram", "interface", "page", "card", "timeline",
                "image", "picture", "frame", "display", "displayed"}
COLOR_WORDS = {"red", "blue", "green", "yellow", "white", "black", "orange", "colored"}
SPATIAL_WORDS = {"left", "right", "above", "below", "behind", "beside", "inside", "outside"}
ACTION_WORDS = {"enter", "entering", "leave", "leaving", "holding", "walking", "crossing"}
ABSTRACT_WORDS = {"concept", "topic", "idea", "abstraction", "engineering", "science",
                  "instructions", "process", "method", "reason", "overview"}
OBJECT_WORDS = {"person", "woman", "man", "dog", "cat", "bicycle", "bike", "bus", "car",
                "ball", "phone", "speaker", "screen", "dashboard", "keyboard", "editor",
                "table", "heading", "slide", "diagram", "map", "button", "control"}
FEATURE_NAMES = (
    "bias", "quoted", "token_count_log", "starts_when", "starts_where", "starts_find",
    "speech_word_count", "visual_word_count", "color_word_count", "spatial_word_count",
    "action_word_count", "abstract_word_count", "object_word_count", "has_while",
    "has_on_screen", "has_spoken_phrase", "question_mark", "speech_and_visual",
)


def words(query: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", query.casefold())


def text_features(query: str) -> np.ndarray:
    tokens = words(query)
    token_set = set(tokens)
    speech = sum(token in SPEECH_WORDS for token in tokens)
    visual = sum(token in VISUAL_WORDS for token in tokens)
    return np.asarray([
        1.0, float(bool(re.search(r"['\"]\S.+?['\"]", query))), math.log1p(len(tokens)),
        float(tokens[:1] == ["when"]), float(tokens[:1] == ["where"]),
        float(tokens[:1] == ["find"]), speech, visual,
        sum(token in COLOR_WORDS for token in tokens),
        sum(token in SPATIAL_WORDS for token in tokens),
        sum(token in ACTION_WORDS for token in tokens),
        sum(token in ABSTRACT_WORDS for token in tokens),
        sum(token in OBJECT_WORDS for token in tokens), float("while" in token_set),
        float("screen" in token_set and ("on" in token_set or "shown" in token_set)),
        float(bool(token_set & {"narrator", "lecturer", "presenter", "speaker"}) and speech > 0),
        float("?" in query), float(speech > 0 and visual > 0),
    ], dtype=np.float64)


def heuristic_route(query: str) -> tuple[str, float, list[str]]:
    tokens = set(words(query))
    speech = sorted(tokens & SPEECH_WORDS)
    visual = sorted(tokens & (VISUAL_WORDS | COLOR_WORDS | SPATIAL_WORDS))
    mixed = bool(speech) and ("while" in tokens or bool(visual))
    if mixed:
        return "HYBRID", 0.95, ["speech_intent", "visual_context"]
    if speech:
        return "SPEECH", 0.95, ["speech_intent"]
    if visual or tokens & (OBJECT_WORDS | ACTION_WORDS):
        return "VISUAL", 0.85, ["visual_intent"]
    return "HYBRID", 0.5, ["ambiguous"]


def fit_softmax(features: np.ndarray, labels: np.ndarray, regularization: float,
                iterations: int = 2000) -> dict:
    if features.ndim != 2 or len(features) != len(labels) or regularization <= 0:
        raise ValueError("invalid routing training data")
    if set(labels.tolist()) != set(range(len(ROUTES))):
        raise ValueError("all routing classes are required")
    mean, scale = features.mean(axis=0), features.std(axis=0)
    mean[0], scale[0] = 0.0, 1.0
    scale[scale < 1e-12] = 1.0
    x = (features - mean) / scale
    weights = np.zeros((features.shape[1], len(ROUTES)), dtype=np.float64)
    counts = np.bincount(labels, minlength=len(ROUTES))
    sample_weights = len(labels) / (len(ROUTES) * counts[labels])
    learning_rate = 0.08
    target = np.eye(len(ROUTES))[labels]
    for _ in range(iterations):
        logits = x @ weights
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        gradient = x.T @ ((probabilities - target) * sample_weights[:, None]) / len(labels)
        gradient += regularization * weights / len(labels)
        gradient[0] -= regularization * weights[0] / len(labels)
        weights -= learning_rate * gradient
    return {"feature_names": FEATURE_NAMES, "routes": ROUTES, "mean": mean.tolist(),
            "scale": scale.tolist(), "weights": weights.tolist(),
            "regularization": regularization, "parameters": int(weights.size)}


def predict_softmax(model: dict, features: np.ndarray) -> np.ndarray:
    x = (features - np.asarray(model["mean"])) / np.asarray(model["scale"])
    logits = x @ np.asarray(model["weights"])
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def learned_route(query: str, model: dict, fallback_threshold: float | None = None):
    probabilities = predict_softmax(model, text_features(query)[None, :])[0]
    index = int(np.argmax(probabilities))
    route, confidence = ROUTES[index], float(probabilities[index])
    if fallback_threshold is not None and confidence < fallback_threshold:
        return "HYBRID", confidence
    return route, confidence
