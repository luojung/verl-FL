"""Read-only stock single-rank FSDP checkpoint layout check, NOT a reload test.

Does not deserialize untrusted pickle files or certify training/model quality.
"""

import argparse
import json
from pathlib import Path


def verify(output, kind, step):
    if kind not in {"sft", "grpo_hf"} or step < 1:
        raise ValueError("kind must be sft/grpo_hf and step must be positive")
    root = output.expanduser().resolve()
    directory = root / f"global_step_{step}"
    model_dir = directory if kind == "sft" else directory / "actor"
    files = [model_dir / f"{prefix}_world_size_1_rank_0.pt" for prefix in ("model", "optim", "extra_state")]
    files += [model_dir / "fsdp_config.json", directory / "data.pt", root / "latest_checkpointed_iteration.txt"]
    for path in files:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty checkpoint artifact: {path}")
    metadata = json.loads((model_dir / "fsdp_config.json").read_text(encoding="utf-8"))
    if metadata.get("world_size") != 1 or metadata.get("FSDP_version") != 1:
        raise ValueError(f"Expected single-rank FSDP1 metadata, got {metadata}")
    recorded_step = int((root / "latest_checkpointed_iteration.txt").read_text(encoding="utf-8").strip())
    if recorded_step != step:
        raise ValueError(f"Tracker records step {recorded_step}, not requested step {step}")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kind", choices=["sft", "grpo_hf"], required=True)
    parser.add_argument("--step", type=int, default=1)
    args = parser.parse_args()
    files = verify(args.output_dir, args.kind, args.step)
    for path in files:
        print(f"Present, nonempty: {path}")
    print(
        "Layout check only: does NOT establish successful deserialization, "
        "reload equivalence, finite loss, or clean process teardown."
    )
    print("SFT uses stock FSDP checkpoints; no adapter-only export is assumed.")


if __name__ == "__main__":
    main()
