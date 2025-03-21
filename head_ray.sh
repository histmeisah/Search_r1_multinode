source /home/ma-user/modelarts/work/jjw/Search-R1/r1_env.sh
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DATA_DIR='/home/ma-user/modelarts/work/jjw/Search-R1/data/nq_search'
# export HYDRA_FULL_ERROR=1
export WANDB_API_KEY='5d830c409e2aa7dff34c333a2f79798a877bfc7b'
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
WAND_PROJECT='Search-R1'
export GLOO_SOCKET_IFNAME=enp218s0   # 多机之间使用gloo通信时需要指定网口名称，
export TP_SOCKET_IFNAME=enp218s0     # 多机之间使用TP通信时需要指定网口名称
export HCCL_SOCKET_IFNAME=enp218s0 
export HYDRA_FULL_ERROR=1
# export BASE_MODEL='/home/ma-user/modelarts/work/jjw/Search-R1/model/meta-llama/Llama-3.2-3B'
export BASE_MODEL='/home/ma-user/modelarts/work/model/Qwen2.5-7B-Instruct'
export EXPERIMENT_NAME=nq-search-r1-ppo-llama3.2-3b-em

export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues

PET_MASTER_PORT=8266
MASTER_ADDR="192.168.154.83"
ray start --head --port=$PET_MASTER_PORT