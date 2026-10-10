#!/usr/bin/env bash
# Shared setup only; sourcing this file does not allocate or launch anything.

init_example() {
    local kind="$1"
    EXAMPLE_KIND="$kind"
    EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd -- "$EXAMPLE_DIR/../.." && pwd)"
    cd -- "$REPO_ROOT"
    : "${MODEL_PATH:?Set MODEL_PATH to an already downloaded local Hugging Face model directory}"
    export MODEL_PATH
    export MODEL_FAMILY="${MODEL_FAMILY:-qwen}"
    case "$MODEL_FAMILY" in
        qwen) TARGET_MODULES=all-linear ;;
        medgemma) TARGET_MODULES=medgemma-language-linear ;;
        *) printf 'MODEL_FAMILY must be qwen or medgemma\n' >&2; return 2 ;;
    esac
    export TRAIN_FILE="${TRAIN_FILE:-$REPO_ROOT/data/flagos_single_card/train.parquet}"
    export VAL_FILE="${VAL_FILE:-$REPO_ROOT/data/flagos_single_card/val.parquet}"
    export OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/flagos_single_card/$MODEL_FAMILY/$kind}"
    export RUN_STEPS="${RUN_STEPS:-1}"
    export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
    export LORA_RANK="${LORA_RANK:-8}"
    export ROLLOUT_N="${ROLLOUT_N:-2}"
    export RAY_NUM_CPUS="${RAY_NUM_CPUS:-4}"
    PYTHON_BIN="${PYTHON_BIN:-python}"
    DRY_RUN="${DRY_RUN:-0}"
    # Resolve user paths once before Ray workers or Hydra change working dirs.
    local path_key path_value
    for path_key in MODEL_PATH TRAIN_FILE VAL_FILE OUTPUT_DIR; do
        path_value="$("$PYTHON_BIN" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "${!path_key}")"
        printf -v "$path_key" '%s' "$path_value"
        export "$path_key"
    done
    if [[ -n "${FLAGOS_BACKEND_CONFIG:-}" ]]; then
        path_value="$("$PYTHON_BIN" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$FLAGOS_BACKEND_CONFIG")"
        export FLAGOS_BACKEND_CONFIG="$path_value"
    fi
    local key
    local numeric_keys=(RUN_STEPS TRAIN_BATCH_SIZE LORA_RANK)
    if [[ "$kind" == grpo_hf ]]; then
        numeric_keys+=(ROLLOUT_N RAY_NUM_CPUS)
    fi
    for key in "${numeric_keys[@]}"; do
        if [[ ! "${!key}" =~ ^[1-9][0-9]*$ ]]; then
            printf '%s must be a positive integer\n' "$key" >&2; return 2
        fi
    done
    if [[ "$kind" == grpo_hf && "$RAY_NUM_CPUS" -lt 4 ]]; then
        printf 'RAY_NUM_CPUS must be >= 4 (TaskRunner plus the 3-CPU worker bundle); allocate those CPUs first.\n' >&2
        return 2
    fi
    if [[ "$kind" == grpo_hf && "$ROLLOUT_N" -lt 2 ]]; then
        printf 'ROLLOUT_N must be >= 2 for GRPO groups\n' >&2; return 2
    fi
    if [[ "$DRY_RUN" != 0 && "$DRY_RUN" != 1 ]]; then
        printf 'DRY_RUN must be 0 or 1\n' >&2; return 2
    fi
    if [[ -n "${FLAGOS_VISIBLE_DEVICES_ENV:-}" && "${FLAGOS_VISIBLE_DEVICES_ENV}" != CUDA_VISIBLE_DEVICES ]]; then
        printf 'Unset FLAGOS_VISIBLE_DEVICES_ENV: this adapter requires canonical CUDA visibility.\n' >&2
        return 2
    fi
    # Preserve the caller/scheduler CUDA/HIP/ROCR assignments. Never force device 0.
    export VERL_PLATFORM=flagos
    export PYTHONUNBUFFERED=1
    export TOKENIZERS_PARALLELISM=false
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
}

hydra_string() {
    "$PYTHON_BIN" -c 'import json, sys; print(json.dumps(sys.argv[1]))' "$1"
}

preflight() {
    if [[ "$DRY_RUN" == 1 ]]; then
        printf 'DRY_RUN: command rendering only; no dependency, data, accelerator, or config-composition validation.\n'
        return
    fi
    "$PYTHON_BIN" - "$EXAMPLE_KIND" <<'PY'
import os
import sys
from pathlib import Path

for key in ("MODEL_PATH", "TRAIN_FILE", "VAL_FILE"):
    path = Path(os.environ[key]).expanduser()
    if not (path.is_dir() if key == "MODEL_PATH" else path.is_file()):
        raise FileNotFoundError(f"{key} must reference an existing local {'directory' if key == 'MODEL_PATH' else 'file'}: {path}")
if not (Path(os.environ["MODEL_PATH"]).expanduser() / "config.json").is_file():
    raise FileNotFoundError("MODEL_PATH must contain Hugging Face config.json")
output = Path(os.environ["OUTPUT_DIR"]).expanduser()
if output.exists() and any(output.iterdir()):
    raise FileExistsError("Use a new or empty OUTPUT_DIR; smoke runs do not overwrite or resume existing artifacts")
route = os.environ.get("FLAGOS_BACKEND_CONFIG")
if route and not Path(route).is_file():
    raise FileNotFoundError("FLAGOS_BACKEND_CONFIG must reference an existing runtime routing configuration")
if sys.argv[1] == "grpo_hf":
    if os.environ.get("RAY_ADDRESS"):
        raise ValueError("Unset RAY_ADDRESS: GRPO starts a local Ray instance within the existing allocation")
    if any(value for key, value in os.environ.items() if key.startswith("RAY_EXPERIMENTAL_NOSET_") and key.endswith("_VISIBLE_DEVICES")):
        raise ValueError("Unset Ray NOSET visibility flags; GRPO requires normal per-actor isolation")

import torch
import torch_fl  # noqa: F401 -- registers PrivateUse1 and the distributed backend

if not torch.flagos.is_available() or torch.flagos.device_count() != 1:
    raise RuntimeError("Run only inside an already allocated, isolated single-accelerator environment (exactly one visible FlagOS device)")
print("Preflight: one visible FlagOS device. Allocation/visibility remain the caller's responsibility.")
print("Runtime routing configuration:", route or "installed Torch-FL default (not asserted to be boxing-only)")
PY
}

launch_example() {
    printf 'Command: '; printf '%q ' "$@"; printf '\n'
    if [[ "$DRY_RUN" == 1 ]]; then
        return
    fi
    mkdir -p -- "$OUTPUT_DIR"
    # pipefail in the caller preserves Python/torchrun failures, including teardown.
    # No os._exit, process kill, synthetic result.json, scheduler, or auto install.
    "$@" 2>&1 | tee "$OUTPUT_DIR/train.log"
}
