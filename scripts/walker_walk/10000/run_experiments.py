#!/usr/bin/env python3
import subprocess
import os
import sys
from typing import List

def run_experiment(seed: int, teacher_betas: List[float]):
    """Run a single experiment with the given seed and teacher betas."""
    cmd = [
        "python", "train_PEBBLE_mixture.py",
        "env=walker_walk",
        f"seed={seed}",
        "agent.params.actor_lr=0.0005",
        "agent.params.critic_lr=0.0005",
        "num_train_steps=500000",
        "agent.params.batch_size=1024",
        "double_q_critic.params.hidden_dim=1024",
        "double_q_critic.params.hidden_depth=2",
        "diag_gaussian_actor.params.hidden_dim=1024",
        "diag_gaussian_actor.params.hidden_depth=2",
        "num_unsup_steps=9000",
        "reward_batch=20scripts/walker_walk/1000/run_pebble_mixture_b[1,1,0,0,0].py0",  # Changed from 10 to 20
        "num_interact=20000",
        "max_feedback=10000",  # Changed from 500 to 1000
        "feed_type=6",
        "reward_update=50",
        "reset_update=100",
        f"teacher_betas={teacher_betas}"
    ]
    
    print(f"Running experiment with seed={seed}, teacher_betas={teacher_betas}")
    result = subprocess.run(cmd, preexec_fn=os.setpgrp, check=True)
    return result.returncode == 0

def run_multiple_seeds(teacher_betas: List[float], seeds: List[int] = [12345, 23451, 34512, 45123, 51234]):
    """Run experiments with multiple seeds for given teacher betas."""
    success = True
    
    for seed in seeds:
        try:
            if not run_experiment(seed, teacher_betas):
                print(f"Experiment failed for seed {seed}")
                success = False
        except subprocess.CalledProcessError as e:
            print(f"Experiment failed for seed {seed}: {e}")
            success = False
    
    return success

if __name__ == "__main__":
    # Get teacher_betas from command line if provided, otherwise use default
    if len(sys.argv) > 1:
        teacher_betas = eval(sys.argv[1])  # e.g., "[1,1,-1,-1,-1]"
    else:
        print("Please provide teacher_betas as command line argument")
        print("Example: ./run_experiments.py '[1,1,-1,-1,-1]'")
        sys.exit(1)
        
    success = run_multiple_seeds(teacher_betas)
    sys.exit(0 if success else 1)