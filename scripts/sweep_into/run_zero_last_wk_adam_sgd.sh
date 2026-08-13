#!/usr/bin/env bash
# Zero-last TTP + two-path w_k on Sweep-Into (10 seeds).
# Reward ensemble: Adam lr=0.0003
# Trust α: SGD lr=0.005
#
# Run from repository root:
#   bash scripts/sweep_into/run_zero_last_wk_adam_sgd.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

for seed in 12345 23451 34512 45123 51234 67890 78906 89067 90678 6789; do
    python train_PEBBLE_mixture_zero_last_wk_adam_sgd.py \
        env=metaworld_sweep-into-v2 \
        seed=$seed \
        agent.params.actor_lr=0.0003 \
        agent.params.critic_lr=0.0003 \
        activation=tanh \
        num_unsup_steps=9000 \
        num_train_steps=1000000 \
        agent.params.batch_size=512 \
        double_q_critic.params.hidden_dim=256 \
        double_q_critic.params.hidden_depth=3 \
        diag_gaussian_actor.params.hidden_dim=256 \
        diag_gaussian_actor.params.hidden_depth=3 \
        reward_update=10 \
        num_interact=5000 \
        max_feedback=40000 \
        reward_batch=50 \
        feed_type=6 \
        teacher_betas=[1,1,1,-1] \
        reward_lr=0.0003 \
        alpha_lr=0.005
done
