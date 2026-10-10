#!/usr/bin/env bash
# Run only within an already allocated single-card environment. See README.md.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
init_example sft
preflight

# Stock SFT FSDP1 + LoRA; keep its normal wrapping/checkpoint implementation.
# Language-only MedGemma LoRA leaves base/vision weights frozen via PEFT.
launch_example "$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 \
    -m verl.trainer.fsdp_sft_trainer \
    "model.partial_pretrain=$(hydra_string "$MODEL_PATH")" \
    model.strategy=fsdp model.fsdp_config.model_dtype=bf16 \
    +model.attn_implementation=eager +model.local_files_only=true \
    model.trust_remote_code=false model.use_liger=false \
    model.enable_gradient_checkpointing=true \
    "model.lora_rank=$LORA_RANK" model.lora_alpha=16 "model.target_modules=$TARGET_MODULES" \
    model.fsdp_config.cpu_offload=false model.fsdp_config.offload_params=false \
    "data.train_files=$(hydra_string "$TRAIN_FILE")" "data.val_files=$(hydra_string "$VAL_FILE")" \
    data.prompt_key=question data.response_key=answer \
    "data.train_batch_size=$TRAIN_BATCH_SIZE" data.micro_batch_size_per_gpu=1 \
    data.max_length=256 data.truncation=error +data.num_workers=0 +data.pin_memory=false \
    use_remove_padding=false ulysses_sequence_parallel_size=1 \
    optim.lr=1e-5 optim.lr_warmup_steps_ratio=0 \
    optim.override_optimizer_config='{foreach:false,fused:false}' \
    trainer.device=flagos trainer.nnodes=1 trainer.n_gpus_per_node=1 \
    trainer.total_epochs=1 "trainer.total_training_steps=$RUN_STEPS" \
    trainer.logger='[console]' trainer.project_name=flagos-single-card \
    "trainer.experiment_name=$MODEL_FAMILY-sft-smoke" \
    "trainer.default_local_dir=$(hydra_string "$OUTPUT_DIR")" \
    trainer.resume_mode=disable trainer.save_freq=1 trainer.test_freq=1
