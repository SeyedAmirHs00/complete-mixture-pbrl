#!/usr/bin/env bash
# Practical-enhancement ablation on MetaWorld Door-Open (adversarial 3R1A).
# Run from repository root.
#
# Examples:
#   bash scripts/door_open/run_enhancement_ablation.sh
#   bash scripts/door_open/run_enhancement_ablation.sh --dry-run
#   bash scripts/door_open/run_enhancement_ablation.sh --variants full_ttp raw
#   bash scripts/door_open/run_enhancement_ablation.sh --teacher-betas 1 1 1 0

set -euo pipefail
cd "$(dirname "$0")/../.."
python scripts/door_open/run_enhancement_ablation.py "$@"
