#!/usr/bin/env python3
from run_experiments import run_multiple_seeds

if __name__ == "__main__":
    teacher_betas = [1, 1, 1, 0, 0]
    success = run_multiple_seeds(teacher_betas)
    exit(0 if success else 1)exp_pebble_mixture_alpha_sum_log_over/walker_walk/max_feedback1000_feed_type6_n20_l50_g1_b[1, 1, 0, 0, 0]_m0_s0_e0