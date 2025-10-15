for seed in 23451 34512 45123 51234; do
    python train_PEBBLE_mixture.py env=walker_walk seed=$seed agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 num_train_steps=500000 agent.params.batch_size=1024 double_q_critic.params.hidden_dim=1024 double_q_critic.params.hidden_depth=2 diag_gaussian_actor.params.hidden_dim=1024 diag_gaussian_actor.params.hidden_depth=2 num_unsup_steps=9000 reward_batch=10 num_interact=20000 max_feedback=500 feed_type=6 reward_update=50 reset_update=100 \
    teacher_betas=[1,1,1,-1,-1]
done

# python train_PEBBLE_mixture.py env=walker_walk seed=12345 agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 num_train_steps=500000 agent.params.batch_size=1024 double_q_critic.params.hidden_dim=1024 double_q_critic.params.hidden_depth=2 diag_gaussian_actor.params.hidden_dim=1024 diag_gaussian_actor.params.hidden_depth=2 num_unsup_steps=9000 reward_batch=10 num_interact=20000 max_feedback=500 feed_type=6 reward_update=50 reset_update=100 \
#     teacher_betas=[1,1,1,-1,-1]