for seed in 23451 34512 45123 51234 67890 78906 89067 90678 6789; do
    python train_RIME.py --env="walker_walk" --seed=$seed --actor_lr=0.0005 --critic_lr=0.0005 --unsup_steps=9000 --steps=500000 --num_interact=20000 --max_feedback="1000" --reward_batch=100 --reward_update=50 --feed_type=$1 --device="cuda:0" --eps_mistake="0.3" --least_reward_update=15 --threshold_variance='kl' --threshold_alpha=0.5 --threshold_beta_init=3.0 --threshold_beta_min=1.0 --eps_skip="0.0" --eps_equal="0.0" --teacher_gamma="1.0" 
done
