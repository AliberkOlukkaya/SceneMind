"""Build and validate the frozen, source-disjoint router query manifest."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROUTES = ("VISUAL", "SPEECH", "HYBRID")
SPLITS = ("train", "validation", "frozen_test")

# These are independent annotation scenarios, not clips or transcripts. Their domains and
# concepts were authored before candidate fitting. They deliberately exclude every existing
# SceneMind media/source identifier and the protected final-acceptance subject.
SOURCE_CARDS = (
    ("py-debug", "train", "software tutorial",
     ("breakpoint gutter", "stack trace panel", "variable inspector", "test summary", "terminal prompt", "call graph", "settings dialog", "red exception banner", "source tree", "profiler chart"),
     ("reproducible bug reports", "mutable default arguments", "isolating a regression", "reading a traceback", "choosing a breakpoint", "mocking external services", "reducing a failing case", "interpreting exit codes", "debugger side effects", "when to add logging")),
    ("bread-workshop", "train", "ordinary instructional video",
     ("bubbled starter jar", "floured worktop", "folded dough", "covered mixing bowl", "scored loaf", "cast iron pot", "cooling rack", "digital scale", "windowpane stretch", "brown crust"),
     ("why fermentation slows", "judging starter readiness", "controlling dough temperature", "avoiding a dense crumb", "resting before shaping", "steam during baking", "measuring hydration", "fixing overproofed dough", "using less yeast", "waiting before slicing")),
    ("telescope-setup", "train", "equipment tutorial",
     ("tripod bubble level", "finder scope", "counterweight shaft", "polar alignment dial", "eyepiece case", "red flashlight", "star map", "mount control pad", "focus knob", "dew shield"),
     ("why alignment matters", "selecting an eyepiece", "preventing lens fog", "balancing the mount", "finding celestial north", "preserving night vision", "choosing magnification", "correcting star drift", "safe solar observing", "packing the optics")),
    ("watercolor-basics", "train", "art lesson",
     ("wet paper sheen", "round brush tip", "mixing palette", "masking tape border", "color swatch row", "two water jars", "granulated wash", "dry brush texture", "paint bloom", "finished landscape"),
     ("why paper weight matters", "mixing a neutral shadow", "keeping colors transparent", "timing a second wash", "lifting unwanted pigment", "avoiding muddy mixtures", "softening a hard edge", "planning negative space", "testing color strength", "letting layers dry")),
    ("home-network", "train", "technical demonstration",
     ("router status lights", "ethernet port labels", "wireless channel graph", "admin login page", "device list", "speed test gauge", "cable tester", "guest network toggle", "firmware progress bar", "floor plan heatmap"),
     ("separating guest devices", "choosing a wireless channel", "why double NAT happens", "placing an access point", "updating firmware safely", "interpreting packet loss", "using a wired backhaul", "setting DNS servers", "finding an address conflict", "resetting without losing settings")),
    ("bike-service", "train", "repair tutorial",
     ("chain wear gauge", "rear derailleur", "brake pad groove", "torque wrench", "cable barrel adjuster", "cassette teeth", "repair stand", "tire sidewall", "spoke key", "lubricant bottle"),
     ("why gears skip", "checking chain wear", "centering a disc brake", "setting saddle height", "finding a slow puncture", "tightening bolts evenly", "choosing chain lubricant", "avoiding crossed gears", "bedding new pads", "testing after a repair")),
    ("coastal-climate", "train", "science lecture",
     ("sea level map", "tide gauge photograph", "temperature anomaly chart", "storm surge diagram", "salt marsh cross-section", "satellite image", "uncertainty bands", "emissions pathway table", "coastline comparison", "risk legend"),
     ("why local sea level differs", "how tides are normalized", "limits of regional forecasts", "the role of land subsidence", "interpreting uncertainty bands", "why wetlands reduce flooding", "choosing a baseline period", "separating weather from climate", "planning under uncertainty", "communicating low probability risks")),
    ("spreadsheet-cleanup", "train", "software tutorial",
     ("duplicate rows", "filter dropdown", "formula error marker", "pivot table", "conditional formatting", "split text dialog", "date column", "named range box", "blank cell selection", "summary chart"),
     ("why dates import incorrectly", "detecting duplicate records", "normalizing category names", "choosing a lookup formula", "handling missing values", "protecting source columns", "checking totals after cleanup", "using relative references", "avoiding silent coercion", "documenting transformations")),
    ("archive-care", "train", "museum presentation",
     ("cotton gloves", "acid-free folder", "humidity meter", "damaged photograph", "storage box label", "soft cleaning brush", "book cradle", "light meter", "sealed display case", "condition report form"),
     ("why gloves are sometimes avoided", "controlling relative humidity", "recording existing damage", "limiting light exposure", "choosing archival enclosures", "handling brittle paper", "responding to mold", "digitizing without stress", "tracking object movement", "prioritizing conservation work")),
    ("guitar-rhythm", "validation", "music lesson",
     ("chord diagram", "right hand close-up", "metronome display", "pick angle", "muted strings", "rhythm notation", "left thumb position", "tempo setting", "practice checklist", "final chord shape"),
     ("why the rhythm sounds rushed", "counting sixteenth notes", "relaxing the fretting hand", "practicing a difficult change", "using a metronome", "keeping unused strings quiet", "building speed gradually", "hearing the backbeat", "recovering after a mistake", "planning short practice sessions")),
    ("orchard-pruning", "validation", "gardening instruction",
     ("crossing branches", "pruning saw", "branch collar", "open canopy", "water sprouts", "disinfectant bottle", "before-and-after tree", "angled cut", "fruit bud cluster", "removed branch pile"),
     ("why winter timing helps", "identifying dead wood", "protecting the branch collar", "limiting cuts in one season", "encouraging air flow", "handling diseased branches", "training a young tree", "avoiding flush cuts", "recognizing fruiting wood", "deciding what to leave")),
    ("database-indexes", "validation", "technical lecture",
     ("B-tree diagram", "query plan table", "latency histogram", "database console", "composite key example", "buffer cache chart", "full scan warning", "index size graph", "execution timeline", "schema sketch"),
     ("why an index can slow writes", "selecting column order", "reading a query plan", "cardinality estimation errors", "when a scan is cheaper", "covering a query", "maintaining index statistics", "cost of unused indexes", "effects of data skew", "testing an indexing change")),
    ("espresso-dialin", "frozen_test", "ordinary instructional video",
     ("coffee dose scale", "grinder adjustment dial", "level grounds basket", "bottomless portafilter", "espresso stream", "shot timer", "pressure gauge", "spent coffee puck", "two tasting cups", "extraction chart"),
     ("why the shot tastes sour", "changing one variable at a time", "choosing the coffee dose", "reading extraction time", "warming the equipment", "distributing grounds evenly", "responding to channeling", "adjusting for older beans", "balancing bitterness", "recording a recipe")),
    ("wetland-survey", "frozen_test", "field briefing",
     ("survey transect", "water depth ruler", "bird identification card", "sample bottle", "reed bed", "field notebook", "wader boots", "GPS waypoint screen", "weather radar", "habitat map"),
     ("why surveys start early", "avoiding disturbance to nests", "recording uncertain sightings", "choosing sample locations", "cleaning equipment between sites", "estimating group size", "working around changing tides", "reporting an invasive plant", "checking weather risk", "preserving water samples")),
    ("printer-calibration", "frozen_test", "maker tutorial",
     ("first layer test", "filament spool", "nozzle close-up", "bed leveling screen", "temperature tower", "stringing between posts", "slicer preview", "support structure", "warped corner", "finished calibration cube"),
     ("why the first layer lifts", "choosing nozzle temperature", "drying moisture-sensitive filament", "setting retraction distance", "checking extrusion steps", "orienting a strong part", "reducing unnecessary supports", "diagnosing layer shifts", "trading speed for finish", "saving a material profile")),
)

VISUAL_FORMS = (
    "Show me the moment with {v}.", "Where can I see {v}?", "Find {v} in the video.",
    "I need the shot where {v} is visible.", "Jump to {v}.", "Which part has {v} on screen?",
    "Take me to the view of {v}.", "Locate the scene containing {v}.",
    "Can you find a clear view of {v}?", "The bit with {v}.",
    "When is {v} shown most clearly?", "Find the frame that includes {v}.",
    "I'd like to inspect {v}.", "Where does {v} appear?", "Bring up {v}.",
    "Search for the scene with {v}.", "At what point is {v} visible?",
    "Let me see {v}.", "Find the close view of {v}.", "Which timestamp shows {v}?",
    "Go to the picture containing {v}.", "Look for {v} on screen.",
    "Where in the recording is {v}?", "Find a frame with {v}.",
    "Show the section featuring {v}.", "I remember seeing {v}; find it.",
    "Locate {v} visually.", "When does the camera show {v}?",
    "Open the moment that displays {v}.", "The scene where {v} appears.",
)

SPEECH_FORMS = (
    "Where do they explain {s}?", "When is {s} discussed?", "Find the explanation of {s}.",
    "What do they say about {s}?", "Take me to the part about {s}.",
    "I want to hear the reasoning behind {s}.", "Where does the lesson cover {s}?",
    "At what point do we learn about {s}?", "Find where the presenter addresses {s}.",
    "The section dealing with {s}.", "When do they get into {s}?",
    "Locate the spoken advice on {s}.", "Where is the rationale for {s} given?",
    "Jump to the discussion of {s}.", "Can you find their comments on {s}?",
    "Which part answers the question of {s}?", "I missed the explanation about {s}.",
    "Find the point where {s} comes up in the narration.", "When is the idea of {s} introduced?",
    "Search the transcript for the guidance on {s}.", "Where do they unpack {s}?",
    "Take me to their argument about {s}.", "Find the verbal explanation concerning {s}.",
    "What is the speaker's advice on {s}?", "Where can I listen to the part on {s}?",
    "The moment they give reasons for {s}.", "When does the narration turn to {s}?",
    "Find the answer they give about {s}.", "Where is {s} talked through?",
    "Go to the passage that covers {s}.",
)

HYBRID_FORMS = (
    "Find where they explain {s} while {v} is visible.",
    "When do they talk about {s} with {v} on screen?",
    "Show me {v} during the explanation of {s}.",
    "Where is {s} discussed as {v} appears?",
    "I need the part about {s} that also shows {v}.",
    "Jump to the explanation of {s} alongside {v}.",
    "Find the moment combining the commentary on {s} and a view of {v}.",
    "When can I hear about {s} and see {v} at the same time?",
    "Locate {v} in the section where {s} is covered.",
    "The bit where {v} supports what they say about {s}.",
    "Show the scene with {v} as the presenter addresses {s}.",
    "Which timestamp pairs {v} with the discussion of {s}?",
    "Find their advice on {s} while the video displays {v}.",
    "Where do the narration about {s} and {v} overlap?",
    "Take me to {v} in the spoken section on {s}.",
    "I remember {v} being shown during the point about {s}.",
    "When does the visual of {v} accompany the explanation of {s}?",
    "Search for {s}, specifically the moment with {v}.",
    "Find the combined visual and spoken example: {v} with {s}.",
    "Where is {v} used while they reason about {s}?",
    "Bring up the part that shows {v} and describes {s}.",
    "At what point does {v} appear during the guidance on {s}?",
    "Find the explanation around {s}; {v} should be visible too.",
    "Let me see {v} while listening to the section about {s}.",
    "Where do they connect {s} to the displayed {v}?",
    "The moment featuring {v} and the narration on {s}.",
    "Find {v}, but only where the speaker is covering {s}.",
    "Which scene shows {v} as they answer the question about {s}?",
    "Go to the visual example {v} in the explanation of {s}.",
    "Locate the passage on {s} that is illustrated by {v}.",
)


def _tags(query: str, route: str, source_index: int, form_index: int) -> list[str]:
    tags = []
    tokens = re.findall(r"[a-z0-9]+", query.casefold())
    if query.endswith("?"):
        tags.append("natural_question")
    if len(tokens) >= 14:
        tags.append("long_query")
    if source_index in {0, 2, 4, 6, 7, 11, 14}:
        tags.append("technical_terms")
    if route == "SPEECH" and not set(tokens) & {"say", "talk", "explain", "discuss", "speaker", "spoken"}:
        tags.append("indirect_speech_intent")
    if route == "VISUAL" and not set(tokens) & {"show", "see", "visible", "screen", "frame", "visual"}:
        tags.append("indirect_visual_intent")
    if route == "HYBRID":
        tags.append("mixed_modality")
    if form_index in {9, 13, 20, 25}:
        tags.append("ambiguous_wording")
    if (route == "SPEECH" and form_index in {4, 8, 15, 23}) or (
        route == "VISUAL" and form_index in {5, 10, 27}
    ):
        tags.append("lexical_overlap_trap")
    return tags or ["direct_intent"]


def build_manifest() -> dict:
    rows = []
    sources = []
    for source_index, (source_id, split, domain, visuals, speech) in enumerate(SOURCE_CARDS):
        sources.append({"source_id": source_id, "source_group": f"erg-{source_id}",
                        "split": split, "domain": domain,
                        "provenance": "independently authored annotation scenario; no media ingested"})
        offset = (source_index * 7) % 30
        for local_index in range(10):
            form_index = (offset + local_index * 3) % 30
            values = {
                "VISUAL": VISUAL_FORMS[form_index].format(v=visuals[local_index]),
                "SPEECH": SPEECH_FORMS[(form_index + 1) % 30].format(s=speech[local_index]),
                "HYBRID": HYBRID_FORMS[(form_index + 2) % 30].format(
                    s=speech[(local_index + 3) % 10], v=visuals[local_index]),
            }
            for route, query in values.items():
                rows.append({
                    "query_id": f"{source_id}-{route[0].lower()}{local_index + 1:02d}",
                    "source_id": source_id, "source_group": f"erg-{source_id}",
                    "split": split, "domain": domain, "query": query,
                    "route": route,
                    "evidence_required": {
                        "VISUAL": "visual evidence alone identifies the requested moment",
                        "SPEECH": "spoken/transcript evidence alone identifies the requested moment",
                        "HYBRID": "both the spoken point and displayed context are required",
                    }[route],
                    "tags": _tags(query, route, source_index, form_index),
                })
    return {
        "schema_version": "1.0.0", "suite_id": "english-router-generalization-v1",
        "annotation_status": "frozen", "language": "en",
        "prohibited_source_ids": ["final-english-acceptance-v1", "personal-acceptance-v1",
                                  "natural-v2", "query-routing-v1"],
        "sources": sources, "queries": rows,
    }


def canonical_bytes(manifest: dict) -> bytes:
    return (json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            .encode("utf-8"))


def manifest_hash(manifest: dict) -> str:
    return hashlib.sha256(canonical_bytes(manifest)).hexdigest()


def validate_manifest(manifest: dict, protected_queries: list[str] | None = None) -> dict:
    if manifest.get("annotation_status") != "frozen":
        raise ValueError("router test manifest must be frozen")
    sources = manifest.get("sources", [])
    rows = manifest.get("queries", [])
    source_ids = [source["source_id"] for source in sources]
    if len(source_ids) != len(set(source_ids)) or len(source_ids) < 9:
        raise ValueError("independent source IDs are required")
    split_sources = {split: {s["source_id"] for s in sources if s["split"] == split}
                     for split in SPLITS}
    if any(not values for values in split_sources.values()):
        raise ValueError("train, validation and frozen test all require sources")
    if any(split_sources[a] & split_sources[b] for a in SPLITS for b in SPLITS if a != b):
        raise ValueError("source-disjoint split violated")
    source_split = {source["source_id"]: source["split"] for source in sources}
    if any(row["split"] != source_split.get(row["source_id"]) for row in rows):
        raise ValueError("query split does not match its source")
    if len(rows) < 300 or len({row["query"] for row in rows}) != len(rows):
        raise ValueError("at least 300 unique natural queries are required")
    for split in SPLITS:
        counts = Counter(row["route"] for row in rows if row["split"] == split)
        if set(counts) != set(ROUTES) or max(counts.values()) - min(counts.values()) > 1:
            raise ValueError(f"route balance failed for {split}")
    protected = {_normalize(value) for value in protected_queries or []}
    leaked = sorted(row["query_id"] for row in rows if _normalize(row["query"]) in protected)
    if leaked:
        raise ValueError(f"protected acceptance query leakage: {leaked}")
    return {"sha256": manifest_hash(manifest), "sources": len(sources),
            "queries": len(rows), "split_sources": {k: len(v) for k, v in split_sources.items()},
            "split_queries": {split: sum(row["split"] == split for row in rows)
                              for split in SPLITS},
            "split_route_counts": {split: dict(Counter(
                row["route"] for row in rows if row["split"] == split
            )) for split in SPLITS},
            "route_counts": dict(Counter(row["route"] for row in rows))}


def _normalize(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def write_manifest(path: Path) -> dict:
    manifest = build_manifest()
    validate_manifest(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest
