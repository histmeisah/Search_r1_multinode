
source your_env
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DATA_DIR=''

# export HYDRA_FULL_ERROR=1
export WANDB_API_KEY=''
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
# WAND_PROJECT='Search-R1'
export GLOO_SOCKET_IFNAME=enp218s0   # 多机之间使用gloo通信时需要指定网口名称，
export TP_SOCKET_IFNAME=enp218s0     # 多机之间使用TP通信时需要指定网口名称
export HCCL_SOCKET_IFNAME=enp218s0 
export HYDRA_FULL_ERROR=1

# export BASE_MODEL=''
export BASE_MODEL=''
export EXPERIMENT_NAME=nq-search-r1-ppo-llama3.2-3b-em

export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues

PET_MASTER_PORT=
MASTER_ADDR=""
ray start --head --port=$PET_MASTER_PORT

