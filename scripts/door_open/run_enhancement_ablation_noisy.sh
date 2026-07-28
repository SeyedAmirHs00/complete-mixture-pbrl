#!/usr/bin/env bash
# Practical-enhancement ablation on MetaWorld Door-Open (noisy 3R1N).
# Run from repository root.

set -euo pipefail
cd "$(dirname "$0")/../.."
python scripts/door_open/run_enhancement_ablation.py --teacher-betas 1 1 1 0 "$@"
