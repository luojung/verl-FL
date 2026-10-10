#!/usr/bin/env bash
# Schema mode=async is required; the integrated HF route executes synchronously.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
init_example grpo_hf
preflight

runtime_env=("+ray_kwargs.ray_init.runtime_env.env_vars.VERL_PLATFORM=flagos")
# Explicitly forward routing to Ray workers; do not install or alter runtime configs.
if [[ -n "${FLAGOS_BACKEND_CONFIG:-}" ]]; then
    runtime_env+=("+ray_kwargs.ray_init.runtime_env.env_vars.FLAGOS_BACKEND_CONFIG=$(hydra_string "$FLAGOS_BACKEND_CONFIG")")
fi
launch_example "$PYTHON_BIN" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=false \
    "data.train_files=$(hydra_string "$TRAIN_FILE")" "data.val_files=$(hydra_string "$VAL_FILE")" \
    data.prompt_key=prompt "data.train_batch_size=$TRAIN_BATCH_SIZE" data.val_batch_size=2 \
    data.max_prompt_length=128 data.max_response_length=32 data.truncation=error \
    data.dataloader_num_workers=0 data.filter_overlong_prompts=false \
    data.return_multi_modal_inputs=false \
    "actor_rollout_ref.model.path=$(hydra_string "$MODEL_PATH")" \
    +actor_rollout_ref.model.override_config.attn_implementation=eager \
    actor_rollout_ref.model.trust_remote_code=false actor_rollout_ref.model.use_remove_padding=false \
    actor_rollout_ref.model.use_liger=false actor_rollout_ref.model.use_fused_kernels=false \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.enable_activation_offload=false \
    "actor_rollout_ref.model.lora_rank=$LORA_RANK" actor_rollout_ref.model.lora_alpha=16 \
    "actor_rollout_ref.model.target_modules=$TARGET_MODULES" \
    actor_rollout_ref.actor.strategy=fsdp actor_rollout_ref.actor.freeze_vision_tower=true \
    "actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=false actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.calculate_entropy=false actor_rollout_ref.actor.use_torch_compile=false \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.override_optimizer_config='{foreach:false,fused:false}' \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 actor_rollout_ref.actor.fsdp_config.dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=1 actor_rollout_ref.actor.fsdp_config.use_orig_params=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=false \
    actor_rollout_ref.rollout.name=hf actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.dtype=bfloat16 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.do_sample=true actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.top_k=0 "actor_rollout_ref.rollout.n=$ROLLOUT_N" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    reward_model.enable=false reward_model.use_reward_loop=false \
    "custom_reward_function.path=$(hydra_string "$EXAMPLE_DIR/smoke_reward.py")" \
    custom_reward_function.name=compute_score \
    trainer.device=flagos trainer.nnodes=1 trainer.n_gpus_per_node=1 \
    trainer.use_legacy_worker_impl=enable trainer.total_epochs=1 \
    "trainer.total_training_steps=$RUN_STEPS" trainer.val_before_train=false \
    trainer.save_freq=1 trainer.test_freq=1 trainer.logger='[console]' \
    trainer.project_name=flagos-single-card "trainer.experiment_name=$MODEL_FAMILY-grpo-hf-smoke" \
    "trainer.default_local_dir=$(hydra_string "$OUTPUT_DIR")" trainer.resume_mode=disable \
    +ray_kwargs.ray_init.address=local +ray_kwargs.ray_init.num_gpus=1 \
    "ray_kwargs.ray_init.num_cpus=$RAY_NUM_CPUS" "${runtime_env[@]}"
