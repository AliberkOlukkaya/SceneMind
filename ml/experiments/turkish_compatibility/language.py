"""Cheap Turkish normalization, routing features and query adaptation."""

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np

ARTIFACT_PATH = Path(__file__).with_name("turkish_adapter_v1.json")
ROUTES = ("VISUAL", "SPEECH", "HYBRID")
FEATURE_NAMES = (
    "bias",
    "token_count_log",
    "speech_count",
    "visual_count",
    "object_count",
    "connector_count",
    "speech_and_visual",
    "speech_and_connector",
    "question",
    "imperative_find",
    "imperative_show",
    "apostrophe",
    "technical_ascii",
    "ambiguous_part",
)


@lru_cache(maxsize=1)
def artifact() -> dict:
    return json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))


def normalize_turkish(text: str) -> str:
    value = unicodedata.normalize("NFC", text).replace("’", "'").replace("‘", "'")
    value = re.sub(
        r"(?<!\w)[A-Z][A-Z0-9+#.]{1,}(?!\w)",
        lambda match: match.group(0).lower(),
        value,
    )
    value = value.translate(str.maketrans({"I": "ı", "İ": "i"})).lower()
    return " ".join(value.split())


def words(text: str) -> list[str]:
    normalized = normalize_turkish(text)
    return re.findall(r"[^\W_]+(?:'[^\W_]+)?", normalized, flags=re.UNICODE)


def base_token(token: str) -> str:
    return token.split("'", 1)[0]


def _root_count(tokens: list[str], roots: list[str]) -> int:
    return sum(any(base_token(token).startswith(root) for root in roots) for token in tokens)


def route_features(text: str) -> np.ndarray:
    config = artifact()
    tokens = words(text)
    normalized = normalize_turkish(text)
    speech = _root_count(tokens, config["speech_roots"])
    visual = _root_count(tokens, config["visual_roots"])
    objects = _root_count(tokens, config["visual_objects"])
    connectors = sum(connector in normalized for connector in config["hybrid_connectors"])
    technical = sum(bool(re.fullmatch(r"[a-zA-Z0-9+#.]+(?:'[A-Za-z]+)?", raw))
                    for raw in text.split())
    return np.asarray([
        1.0,
        np.log1p(len(tokens)),
        speech,
        visual,
        objects,
        connectors,
        float(speech > 0 and (visual > 0 or objects > 0)),
        float(speech > 0 and connectors > 0),
        float("?" in text),
        float(bool(tokens) and base_token(tokens[0]).startswith("bul")),
        float(bool(tokens) and base_token(tokens[0]).startswith("göster")),
        float("'" in normalized),
        technical,
        float(any(base_token(token) in {"kısım", "bölüm", "yer"} for token in tokens)
              and speech == 0 and visual == 0 and objects == 0),
    ], dtype=np.float64)


def rule_route(text: str) -> tuple[str, list[str]]:
    config = artifact()
    tokens = words(text)
    normalized = normalize_turkish(text)
    speech = _root_count(tokens, config["speech_roots"])
    visual = _root_count(tokens, config["visual_roots"])
    objects = _root_count(tokens, config["visual_objects"])
    connector = any(value in normalized for value in config["hybrid_connectors"])
    if speech and connector:
        return "HYBRID", ["turkish_speech_intent", "visual_context"]
    if speech:
        return "SPEECH", ["turkish_speech_intent"]
    if visual or objects:
        return "VISUAL", ["turkish_visual_intent"]
    return "HYBRID", ["ambiguous"]


def _is_stop(token: str) -> bool:
    root = base_token(token)
    return any(root.startswith(value) for value in artifact()["retrieval_stop_roots"])


def _translate_token(token: str) -> str | None:
    root = base_token(token)
    lexicon = artifact()["translation_lexicon"]
    if root in lexicon:
        return lexicon[root]
    for source, target in lexicon.items():
        if len(source) >= 4 and root.startswith(source):
            return target
    if re.fullmatch(r"[a-z0-9+#.]+", root):
        return root
    return None


def adapt_visual(text: str) -> str:
    translated = [_translate_token(token) for token in words(text) if not _is_stop(token)]
    return " ".join(token for token in translated if token) or normalize_turkish(text)


TURKISH_SUFFIXES = (
    "lerinden", "larından", "lerden", "lardan", "inde", "ında", "unda", "ünde",
    "den", "dan", "ten", "tan", "nin", "nın", "nun", "nün", "yi", "yı", "yu", "yü",
    "i", "ı", "u", "ü",
)


def stem_turkish(token: str) -> str:
    value = base_token(token)
    for suffix in TURKISH_SUFFIXES:
        if len(value) - len(suffix) >= 4 and value.endswith(suffix):
            return value[:-len(suffix)]
    return value


def normalized_bm25_tokens(text: str) -> list[str]:
    return [stem_turkish(token) for token in words(text) if not _is_stop(token)]


def adapt_speech(text: str, transcript_language: str | None) -> str:
    if transcript_language == "en":
        return adapt_visual(text)
    return " ".join(normalized_bm25_tokens(text))
