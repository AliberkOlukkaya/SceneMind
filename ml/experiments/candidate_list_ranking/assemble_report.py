"""Freeze the measured report with the evidence-led architecture decision."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def assemble(source: Path, output: Path) -> dict:
    report = json.loads(source.read_text(encoding="utf-8"))
    report["artifact_inputs"] = {
        "raw_measurement_sha256": hashlib.sha256(source.read_bytes()).hexdigest()
    }
    report["decision"] = {
        "outcome": "E_routing_is_the_dominant_failure",
        "production_promoted": False,
        "recommendation": (
            "Keep the production 5-second CLIP order and explicit visual/speech/hybrid mode. "
            "Do not ship the learned list scorer or no-match threshold. Make channel selection "
            "clearer and evaluate it on new calibration speech sources before considering "
            "automatic deterministic routing. A stronger semantic reranker is justified only "
            "after the route is correct; top-50 oracle headroom alone did not transfer through "
            "the cheap calibration-only scorers."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "ml/evaluation/reports/candidate-list-ranking-v1.json")
    args = parser.parse_args()
    result = assemble(args.source, args.output)
    print(json.dumps(result["decision"], indent=2))

