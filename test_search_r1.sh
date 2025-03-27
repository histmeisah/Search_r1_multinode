#!/bin/bash

# Environment setup
source /home/ma-user/modelarts/work/jjw/Search-R1/r1_env.sh
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DATA_DIR=''
export WANDB_API_KEY=''
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues

# Model configuration
export BASE_MODEL='/home/ma-user/modelarts/work/model/Qwen2.5-7B'
export WANDB_PROJECT='Search-R1'
# 动态生成带有日期的experiment_name
export CURRENT_DATETIME=$(date +"%Y-%m-%d_%H-%M-%S")
export EXPERIMENT_NAME="nq-search-r1-grpo-qwen2.5-7b-${CURRENT_DATETIME}"
# Search-related configuration
export ENABLE_SEARCH=true

export SEARCH_URL=""
export SEARCH_TOPK=3
export MAX_OBS_LENGTH=300
export MAX_TURNS=2

# Training parameters
export BATCH_SIZE=32
export MINI_BATCH_SIZE=16 
export MICRO_BATCH_SIZE=4
export MAX_PROMPT_LENGTH=4096
export MAX_RESPONSE_LENGTH=500
export LR=1e-6

# 设置环境变量以显示完整错误堆栈
export HYDRA_FULL_ERROR=1

# 测试搜索API可用性
echo "Testing search API..."
curl -X POST "$SEARCH_URL" \
  -H "Content-Type: application/json" \
  -d '{"queries": ["test query"], "topk": 3, "return_scores": true}' \
  --connect-timeout 5 \
  --max-time 10 \
  || { echo "Search API not available at $SEARCH_URL"; exit 1; }
echo "Search API test passed!"
sleep 10
# Run the training with environment variables
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=$BATCH_SIZE \
    data.val_batch_size=32 \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.95 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.enable_search=$ENABLE_SEARCH \
    actor_rollout_ref.rollout.search_url=\"$SEARCH_URL\" \
    actor_rollout_ref.rollout.search_topk=$SEARCH_TOPK \
    actor_rollout_ref.rollout.max_turns=$MAX_TURNS \
    actor_rollout_ref.rollout.max_obs_length=$MAX_OBS_LENGTH \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@
