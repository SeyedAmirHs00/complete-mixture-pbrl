#!/bin/bash
# Missing seeds for exp_pebble_mixture - cheetah_run
# Configuration: max_feedback4000_feed_type6_n100_l50_g1_b[1, 1, 1, 0]_m0_s0_e0
# Missing seed: seed51234

cd ../..

for seed in 51234; do
    python train_PEBBLE_mixture.py env=cheetah_run seed=$seed agent.params.actor_lr=0.0003 agent.params.critic_lr=0.0003 num_train_steps=1000000 agent.params.batch_size=512 double_q_critic.params.hidden_dim=256 double_q_critic.params.hidden_depth=3 diag_gaussian_actor.params.hidden_dim=256 diag_gaussian_actor.params.hidden_depth=3 num_unsup_steps=9000 reward_batch=100 num_interact=5000 max_feedback=4000 feed_type=6 reward_update=10 teacher_betas=[1,1,1,0]
done
