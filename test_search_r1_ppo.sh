#!/bin/bash

# Environment setup
source /home/ma-user/modelarts/work/jjw/Search-R1/r1_env.sh
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DATA_DIR='/home/ma-user/modelarts/work/jjw/Search-R1/data/nq_search'
export WANDB_API_KEY='5d830c409e2aa7dff34c333a2f79798a877bfc7b'
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues

# Model configuration
# export BASE_MODEL='/home/ma-user/modelarts/work/model/Qwen2.5-7B'
export BASE_MODEL="/home/ma-user/modelarts/work/model/Meta-Llama-3-8B"
export WANDB_PROJECT='Search-R1'
export EXPERIMENT_NAME='nq-search-r1-ppo-llama3-8b-3-26-1838'

# Search-related configuration
export ENABLE_SEARCH=true
export SEARCH_URL="http://192.168.194.186:8004/retrieve"
export SEARCH_TOPK=3
export MAX_TURNS=5
export IGNORE_EOS=true

# Training parameters
export BATCH_SIZE=128
export MINI_BATCH_SIZE=16
export MICRO_BATCH_SIZE=4
export MAX_PROMPT_LENGTH=4096
export MAX_RESPONSE_LENGTH=1024
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
    algorithm.adv_estimator=gae \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=$BATCH_SIZE \
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
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.enable_search=$ENABLE_SEARCH \
    actor_rollout_ref.rollout.search_url=\"$SEARCH_URL\" \
    actor_rollout_ref.rollout.search_topk=$SEARCH_TOPK \
    actor_rollout_ref.rollout.max_turns=$MAX_TURNS \
    actor_rollout_ref.rollout.ignore_eos=$IGNORE_EOS \
    algorithm.kl_ctrl.kl_coef=0.001 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=$BASE_MODEL \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=32 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@