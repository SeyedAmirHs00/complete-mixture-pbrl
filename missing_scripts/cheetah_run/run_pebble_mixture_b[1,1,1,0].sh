#!/bin/bash
# Missing seeds for exp_pebble_mixture - cheetah_run
# Configuration: max_feedback4000_feed_type6_n100_l50_g1_b[1, 1, 1, 0]_m0_s0_e0
# Missing seed: seed51234

for seed in 45123, 51234; do
    python train_PEBBLE_mixture.py env=cheetah_run seed=$seed agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 num_train_steps=1000000 agent.params.batch_size=1024 double_q_critic.params.hidden_dim=1024 double_q_critic.params.hidden_depth=2 diag_gaussian_actor.params.hidden_dim=1024 diag_gaussian_actor.params.hidden_depth=2 \
    num_unsup_steps=9000 reward_batch=100 num_interact=20000 max_feedback=4000 feed_type=6 reward_update=50 reset_update=100 \
    teacher_betas=[1,1,1,0]
done
