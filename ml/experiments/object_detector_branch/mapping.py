"""Small, auditable natural-language to COCO-class rules."""

import re
import unicodedata

from ml.experiments.object_detector_branch.schema import QueryObjectMapping

MAPPING_REVISION = "coco-aliases-v1"
ROUTABLE_QUERY_TYPES = {"OBJECT", "COMPOSITIONAL", "NEGATIVE"}

# Inner tuples are OR-groups. Multiple groups from one alias are AND requirements.
ALIAS_RULES: tuple[tuple[str, tuple[tuple[str, ...], ...]], ...] = (
    ("mobile phone", (("cell phone",),)),
    ("cellphone", (("cell phone",),)),
    ("smartphone", (("cell phone",),)),
    ("phone", (("cell phone",),)),
    ("sports ball", (("sports ball",),)),
    ("ball", (("sports ball",),)),
    ("bicycle", (("bicycle",),)),
    ("bike", (("bicycle",),)),
    ("cyclist", (("person",), ("bicycle",))),
    ("backpack", (("backpack",),)),
    ("bag", (("backpack", "handbag"),)),
    ("bottle", (("bottle",),)),
    ("cup", (("cup",),)),
    ("city bus", (("bus",),)),
    ("bus", (("bus",),)),
    ("car", (("car",),)),
    ("dog", (("dog",),)),
    ("airplane", (("airplane",),)),
    ("aeroplane", (("airplane",),)),
    ("table", (("dining table",),)),
    ("pedestrian", (("person",),)),
    ("children", (("person",),)),
    ("child", (("person",),)),
    ("woman", (("person",),)),
    ("man", (("person",),)),
    ("person", (("person",),)),
    ("speaker", (("person",),)),
    ("jumper", (("person",),)),
)

RELATION_WORDS = {"beside", "holding", "near", "next to", "on", "riding", "wearing"}
UNSUPPORTED_TERMS = {
    "black shirt", "classroom", "deep hole", "holding", "outdoors", "podium",
    "projected slide", "projection screen", "red", "riding", "title screen", "wearing",
    "yellow",
}


def normalize_query(query: str) -> str:
    text = unicodedata.normalize("NFKC", query).casefold()
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


def map_query(query: str, query_type: str) -> QueryObjectMapping:
    normalized = normalize_query(query)
    padded = f" {normalized} "
    groups: list[list[str]] = []
    aliases: dict[str, list[str]] = {}
    occupied: set[str] = set()
    for alias, alias_groups in ALIAS_RULES:
        if f" {alias} " not in padded:
            continue
        # Prefer the first (longest/most specific) phrase for the same words.
        if any(alias in previous or previous in alias for previous in occupied):
            continue
        occupied.add(alias)
        converted = [list(group) for group in alias_groups]
        for group in converted:
            if group not in groups:
                groups.append(group)
        aliases[alias] = [item for group in converted for item in group]
    enabled = query_type in ROUTABLE_QUERY_TYPES and bool(groups)
    if not enabled:
        groups = []
        aliases = {}
    relationship = any(f" {word} " in padded for word in RELATION_WORDS)
    unsupported = sorted(term for term in UNSUPPORTED_TERMS if f" {term} " in padded)
    return QueryObjectMapping(
        normalized_query=normalized,
        query_type=query_type,
        detector_enabled=enabled,
        required_class_groups=groups,
        matched_aliases=aliases,
        relationship_required=relationship,
        unsupported_terms=unsupported,
    )
