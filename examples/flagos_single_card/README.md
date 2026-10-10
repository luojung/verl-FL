# FlagOS single-card SFT and HF-GRPO smoke examples

These examples target the reviewed branch, not upstream multi-card support.
They contain synthetic arithmetic only: **no private model paths, server names,
credentials, patient data, medical evaluation, or pre-recorded success results**.
Run from the repository root in a **pre-existing environment and an already
allocated, isolated single-accelerator session**. Obtain allocation yourself;
these scripts do not submit scheduler jobs, install packages, or connect remotely.

## Prerequisites and scope

- Installed compatible PyTorch, Torch-FL, Transformers, PEFT, VeRL dependencies;
  GRPO additionally needs Ray. Data preparation needs an already installed
  `pyarrow`. The scripts never install/upgrade any of them.
- An already downloaded local Hugging Face model directory, including weights,
  tokenizer and `config.json`. Required gated-model access must have been handled
  separately. Offline flags prevent implicit Hub downloads; remote code is disabled.
- Exactly one visible `torch.flagos` device. The preflight checks availability and
  count, **not** scheduler allocation ownership or whether other users share it.
- Preserve the allocation's CUDA/HIP/ROCR visibility; **do not blindly set it to
  `0`**. Worker normalizes Ray's HIP/ROCR assignment to `CUDA_VISIBLE_DEVICES`.
  Conflicting `FLAGOS_VISIBLE_DEVICES_ENV` overrides are rejected. **GRPO only**
  rejects Ray NOSET visibility flags and `RAY_ADDRESS`; SFT does not use Ray and
  ignores inherited Ray/rollout settings. GRPO starts local Ray inside your allocation,
  with one GPU resource, rather than attaching to an existing cluster.
- These are text-only, eager-attention BF16 LoRA smoke runs, not full-finetuning,
  multimodal training, or throughput/quality benchmarks. No FlashAttention,
  Liger, compiled actor kernels, parameter/optimizer offload or vLLM is selected.
  Both optimizers explicitly set `foreach=false` and `fused=false` for this
  conservative boxing smoke path.
  Supported sizes still depend on accelerator memory and the runtime build.

HF-GRPO deliberately selects `trainer.use_legacy_worker_impl=enable`, FSDP1,
`reward_model.use_reward_loop=false`, and `rollout.name=hf`. Its schema still
requires `rollout.mode=async`, but the integrated trainer takes the **synchronous
shared-HF route**, not an asynchronous inference engine. The integrated worker
constructs a **single-rank root-only FSDP `NO_SHARD` actor**, sharing its live
weights with generation. Do not change this to FSDP2, nested wrapping, or
multi-card generation and assume the reviewed path still applies.

SFT uses **the stock SFT FSDP1 trainer, wrapping, and checkpoint manager**. It does
not reuse the HF-GRPO root-only wrapping rule. For MedGemma, the integrated
`medgemma-language-linear` LoRA target restricts adaptation to language layers;
PEFT freezes base weights, including vision weights. HF-GRPO also requests
`actor.freeze_vision_tower=true`. No image/video columns are generated. Confirm
actual trainable parameter names in runtime output; model revisions can differ.

## Environment parameters

Relative model/data/output paths are interpreted from the repository root.
Paths are passed as quoted Hydra string values; choose ordinary local paths.

| Variable | Required / default | Meaning |
| --- | --- | --- |
| `MODEL_PATH` | **Required** | Existing local Hugging Face model directory; not a Hub ID. |
| `MODEL_FAMILY` | `qwen` | `qwen` or `medgemma`; selects LoRA targets, not a model download. |
| `TRAIN_FILE` | `data/flagos_single_card/train.parquet` | Local prepared data file. |
| `VAL_FILE` | `data/flagos_single_card/val.parquet` | Local prepared validation file. |
| `OUTPUT_DIR` | `outputs/flagos_single_card/<family>/<sft or grpo_hf>` | Must be new or empty; automatic resume is disabled. |
| `PYTHON_BIN` | `python` | Interpreter from your existing runtime environment. |
| `RUN_STEPS` | `1` | Positive requested training step limit; dataset/epoch bounds still apply. |
| `TRAIN_BATCH_SIZE` | `2` | Positive prompt/training batch size; per-device microbatch is 1. |
| `LORA_RANK` | `8` | Positive rank; zero/full-finetuning is intentionally excluded. |
| `ROLLOUT_N` | `2` | GRPO only: responses per prompt, at least 2; ignored by SFT. |
| `RAY_NUM_CPUS` | `4` | GRPO only: local Ray CPU resources, at least 4 and within the existing allocation; ignored by SFT. |
| `DRY_RUN` | `0` | `1` renders the command only, without importing Torch or starting training. |
| `FLAGOS_BACKEND_CONFIG` | Existing runtime default | Optional existing Torch-FL routing configuration; forwarded to Ray workers. |

### Runtime boxing selection

Eager attention and disabled actor compilation do **not** prove every ATen op
uses boxing. To select boxing, supply the **boxing/vendor-only routing file
supported by your installed Torch-FL build** through `FLAGOS_BACKEND_CONFIG`.
The scripts preserve that choice and forward it explicitly into Ray; they do
not generate route configurations, install libraries, or guess deprecated
force-backend switches. The unset/default route is not asserted to be pure
boxing. Inspect the active route and runtime dispatch logs before attributing
results to boxing, FlagGems, or another backend. Other build-specific routing
flags/configuration are the runtime owner's responsibility.

## Prepare public smoke data

```bash
python examples/flagos_single_card/prepare_smoke_data.py
```

This writes 16 training and 4 validation Parquet rows. Each row supplies SFT
`question`/`answer` fields and GRPO `prompt`, `data_source`, `reward_model`, and
`extra_info` fields. Training and validation arithmetic differ. Existing files
are refused unless `--overwrite` is explicitly supplied. To change location,
use `--output-dir` and set `TRAIN_FILE`/`VAL_FILE` accordingly. This is a pipeline
smoke dataset, not a meaningful generalization benchmark.

## Render or run

First activate your **existing** runtime and allocate/isolate a card. Replace the
public placeholder below with your local model directory. Do not run training
on an unallocated login node. Start with command rendering:

```bash
MODEL_PATH=/path/to/local-model MODEL_FAMILY=qwen DRY_RUN=1 \
  bash examples/flagos_single_card/run_sft.sh
MODEL_PATH=/path/to/local-model MODEL_FAMILY=medgemma DRY_RUN=1 \
  bash examples/flagos_single_card/run_grpo_hf.sh
```

`DRY_RUN` is **not** dependency, hardware, Parquet, Hydra composition, or model
validation. It creates no output directory or success artifact.

After satisfying the prerequisites, run one example in the existing allocation:

```bash
MODEL_PATH=/path/to/local-model MODEL_FAMILY=qwen \
  OUTPUT_DIR=outputs/my-qwen-sft-smoke \
  bash examples/flagos_single_card/run_sft.sh

MODEL_PATH=/path/to/local-medgemma MODEL_FAMILY=medgemma \
  OUTPUT_DIR=outputs/my-medgemma-grpo-smoke \
  bash examples/flagos_single_card/run_grpo_hf.sh
```

Logs go to `OUTPUT_DIR/train.log`. The shell uses `pipefail`, so training,
checkpoint, and ordinary teardown failures remain visible. There is no forced
process termination, scheduler invocation, synthetic `result.json`, or inferred
success from merely reaching a training log line.

## Check checkpoint evidence honestly

Both launchers request a checkpoint each step and console validation. After the
process returns normally, inspect its exit status, losses/rewards, trainable
parameters and logs. The toy exact-integer reward can correctly return **zero**
for all sampled outputs. If all responses within an `n=2` group earn the same
reward (all zero **or** all correct), GRPO advantages can be zero and there is
**no reward-driven policy gradient**. A logged optimizer/training step alone does
not prove a nonzero parameter update, learning, or quality (weight decay can also
move parameters independently). Inspect rewards, advantages, gradient norms and
trusted before/after parameter comparisons. If groups tie, retry with more
diverse responses/prompts or a task-appropriate reward; no update is guaranteed.
This example intentionally avoids artificial reward shaping or fabricated gains.

The optional verifier reads metadata and checks required nonempty stock FSDP
artifacts, the dataloader state and the requested checkpoint tracker:

```bash
python examples/flagos_single_card/verify_checkpoint.py \
  --kind sft --output-dir outputs/my-qwen-sft-smoke --step 1
python examples/flagos_single_card/verify_checkpoint.py \
  --kind grpo_hf --output-dir outputs/my-medgemma-grpo-smoke --step 1
```

SFT expects `global_step_N/model_world_size_1_rank_0.pt`, optimizer/extra state,
`fsdp_config.json`, and `data.pt`. GRPO puts model files under
`global_step_N/actor/`. SFT saves **stock FSDP state**, even with LoRA enabled:
**do not expect an adapter-only Hugging Face export**. Any separate GRPO adapter
export is not verified by this helper. The verifier does not deserialize pickle,
verify tensor contents, establish reload equivalence, test an adapter, or certify
clean teardown/finite loss. A nonempty corrupt artifact can pass this layout
check; perform an explicit trusted-runtime reload as separate evidence.

## 底层运行逻辑与原理分析

Static chain for these examples: synthetic CPU Parquet rows → tokenizer/chat
formatting → text tensors on `flagos` → eager HF forward (and synchronous HF
generation for GRPO) → loss/reward/LoRA gradients → stock FSDP checkpoint transfer.
Torch-FL registers the renamed PrivateUse1 device and chooses ATen implementations
through its active routing configuration. BF16 and eager attention do not by
themselves prove kernel identity, fusion, or absence of host/device copies.

The HF path's root-only `NO_SHARD` constraint prevents nested/sharded FSDP
`generate()` misuse; it is not a general multi-card fix. Hardware execution,
ATen overload/dispatcher routes, streams, copies, collective behavior, fusion,
checkpoint reload and process teardown still require fresh runtime evidence.
These examples and their static checks are not new hardware validation; the
historical single-card logs must not be described as a run of these files.

Local development checks cover Bash syntax, four model-family command-rendering
variants, production YAML key inventories, and dependency-isolated helper logic.
These are **not full Hydra composition or dataclass validation**; the review
interpreter lacks Hydra/Torch/Ray/Transformers/pyarrow. Validate actual composed
configs and model/checkpoint behavior in the intended existing runtime before
claiming a reproduction. No new accelerator execution was performed here.

See the parent review document:
[Single-card review (2026-10-10)](../../docs/flagos/single_card_review_20261010.md).

### Parent final-runtime verification (2026-10-10)

After the local checks above, the final source snapshot was tested in an
allocated compute-node CPU environment with real Hydra 1.3.2. All four rendered
Qwen/MedGemma x SFT/HF-GRPO override sets composed successfully.
Run: `20261010-182456-672-verl-final-cpu-regressio` (exit 0).
This closes the example **composition** check only: it does not instantiate all
configuration dataclasses, run the full trainer, exercise DCU, or verify the
example's produced Parquet/checkpoints. Real tiny-model CPU training/export and
HF rollout checks are documented separately in the parent review report.
