We reimplement the Search-r1 base on the newest verl.

We create a new vllm_rollout: `/verl/workers/rollout/vllm_rollout/search_vllm_rollout.py` . This rollout can generation answer and use search engine which developed by search-r1.

So the edited files are:
Train related
1. create search_vllm_rollout.py
2. modify /verl/trainer/main_ppo.py
3. modify verl/trainer/ppo/ray_trainer.py
4. modify verl/trainer/config/ppo_trainer.yaml
Reward related :
1. modify verl/utils/reward_score/__init__.py
2. modify verl/utils/reward_score/qa_em.py
3. modify verl/workers/reward_manager/naive.py



