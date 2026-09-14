"""Freeze the routing report with the measured production decision."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def assemble(source: Path, output: Path) -> dict:
    report = json.loads(source.read_text(encoding="utf-8"))
    if not report["gate"]["passed"] or not report["second_frozen_run"]["passed"]:
        raise ValueError("AUTO cannot be promoted without both frozen passes")
    report["artifact_inputs"] = {
        "raw_measurement_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "router_artifact_sha256": hashlib.sha256(
            (ROOT / "backend/app/query_router_v1.json").read_bytes()
        ).hexdigest(),
    }
    report["production"] = {
        "auto_integrated": True, "auto_default_in_ui": True,
        "feature_flag": "SCENEMIND_AUTO_ROUTING_ENABLED",
        "feature_flag_default": True, "explicit_modes_preserved": True,
        "global_no_match_reopened": False,
    }
    report["decision"] = {
        "outcome": "C_learned_lightweight_routing_wins",
        "production_promoted": True,
        "recommendation": (
            "Ship the 54-parameter text-only AUTO router while retaining Visual, Speech, "
            "and Hybrid overrides. AUTO matches oracle explicit-mode R@5 on the frozen "
            "held-out set. Require a ready transcript for speech-routed searches and treat "
            "the next 30-60 minute personal-video acceptance run as the blocker before v1.0."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "ml/evaluation/reports/query-routing-v1.json")
    args = parser.parse_args()
    print(json.dumps(assemble(args.source, args.output)["decision"], indent=2))
