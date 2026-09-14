"""Deterministic local AUTO search routing using a tiny frozen linear model."""

import json
import math
import re
from functools import lru_cache
from pathlib import Path

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


@lru_cache(maxsize=1)
def artifact() -> dict:
    value = json.loads(Path(__file__).with_name("query_router_v1.json").read_text(encoding="utf-8"))
    if tuple(value["feature_names"]) != FEATURE_NAMES or tuple(value["routes"]) != ROUTES:
        raise ValueError("AUTO routing artifact schema mismatch")
    return value


def auto_route(query: str) -> dict:
    model = artifact()
    features = text_features(query)
    standardized = (features - np.asarray(model["mean"])) / np.asarray(model["scale"])
    logits = standardized @ np.asarray(model["weights"])
    logits -= logits.max()
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()
    index = int(np.argmax(probabilities))
    route = ROUTES[index]
    confidence = float(probabilities[index])
    fallback = model["fallback_threshold"]
    if confidence < fallback:
        route = "HYBRID"
    return {"route": route.lower(), "confidence": confidence,
            "reason": "learned_text_router" if confidence >= fallback else "low_confidence_hybrid"}


def resolve_mode(requested_mode: str, query: str, enabled: bool) -> dict:
    if requested_mode != "auto":
        return {"route": requested_mode, "confidence": None, "reason": "explicit_override"}
    if not enabled:
        return {"route": "hybrid", "confidence": None, "reason": "auto_disabled_fallback"}
    return auto_route(query)
