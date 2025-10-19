#!/usr/bin/env python3
from run_experiments import run_multiple_seeds

if __name__ == "__main__":
    teacher_betas = [1, 1, 0, 0, 0]
    success = run_multiple_seeds(teacher_betas)
    exit(0 if success else 1)