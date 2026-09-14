"""Re-export pinned YOLOX-Nano weights at a fixed ablation input size."""

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

REVISION = "e1052df71842031413f6030723c3607b839c80ce"
CHECKPOINT_SHA256 = "cd28f55fbbc1829f99d9ac9b38a16d259a22889739c8728ea877610201feff7b"
EXPECTED_EXPORTS = {
    640: "a78d8834e54f709b15269c31e4ab4970c8faba014e895814d964bacf0ba4197f",
    768: "aad6f7084d454fd1ab24378e256ee86eddb36e782e91e2f44907c6438d3ae5d3",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(source: Path, checkpoint: Path, output: Path, size: int) -> None:
    if size not in EXPECTED_EXPORTS:
        raise ValueError("only frozen 640 and 768 exports are supported")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source, check=True, capture_output=True, text=True
    ).stdout.strip()
    if revision != REVISION or sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("YOLOX source revision or checkpoint checksum mismatch")
    sys.path.insert(0, str(source))
    import torch
    from torch import nn
    from yolox.exp import get_exp
    from yolox.models.network_blocks import SiLU
    from yolox.utils import replace_module

    experiment = get_exp(str(source / "exps/default/nano.py"), None)
    model = experiment.get_model().eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model = replace_module(model, nn.SiLU, SiLU)
    model.head.decode_in_inference = False
    torch.onnx.export(
        model, torch.randn(1, 3, size, size), str(output), input_names=["images"],
        output_names=["output"], opset_version=11, dynamo=False,
    )
    if sha256(output) != EXPECTED_EXPORTS[size]:
        raise ValueError("export checksum differs from the frozen artifact")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, choices=[640, 768], required=True)
    args = parser.parse_args()
    export(args.source, args.checkpoint, args.output, args.size)
