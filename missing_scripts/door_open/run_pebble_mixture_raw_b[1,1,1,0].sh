#!/bin/bash
# Missing seeds for exp_pebble_mixture_raw - metaworld_door-open-v2
# Configuration: max_feedback40000_feed_type6_n50_l50_g1_b[1, 1, 1, 0]_m0_s0_e0
# Missing seeds: seed89067, seed90678, seed6789

cd ../..

for seed in 89067 90678 6789; do
    python train_PEBBLE_mixture_raw.py env=metaworld_door-open-v2 seed=$seed agent.params.actor_lr=0.0003 agent.params.critic_lr=0.0003 activation=tanh num_unsup_steps=9000 num_train_steps=1000000 agent.params.batch_size=512 double_q_critic.params.hidden_dim=256 double_q_critic.params.hidden_depth=3 diag_gaussian_actor.params.hidden_dim=256 diag_gaussian_actor.params.hidden_depth=3 reward_update=10 num_interact=5000 max_feedback=40000 reward_batch=50 feed_type=6
done
