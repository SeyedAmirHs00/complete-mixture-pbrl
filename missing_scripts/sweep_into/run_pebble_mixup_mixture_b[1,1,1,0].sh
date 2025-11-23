#!/bin/bash
# Missing seeds for exp_pebble_mixture_mixup - metaworld_sweep-into-v2
# Configuration: max_feedback40000_feed_type6_n50_l50_g1_b[1, 1, 1, 0]_m0_s0_e0
# Missing seeds: seed78906, seed89067, seed90678, seed6789

for seed in 67890 78906 89067 90678 6789; do
    python train_PEBBLE_mixup_mixture.py env=metaworld_sweep-into-v2 seed=$seed agent.params.actor_lr=0.0003 agent.params.critic_lr=0.0003 activation=tanh num_unsup_steps=9000 num_train_steps=1000000 agent.params.batch_size=512 double_q_critic.params.hidden_dim=256 double_q_critic.params.hidden_depth=3 diag_gaussian_actor.params.hidden_dim=256 diag_gaussian_actor.params.hidden_depth=3 reward_update=10  num_interact=5000 mixup_alpha=0.5 max_feedback=40000 reward_batch=50  feed_type=6 teacher_betas=[1,1,1,0]
done
