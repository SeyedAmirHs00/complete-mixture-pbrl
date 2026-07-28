#!/usr/bin/env bash
# Plot Door-Open practical-enhancement ablation results (success rate).
# Run from repository root after experiments finish.

set -euo pipefail
cd "$(dirname "$0")/../.."
python plot_enhancement_ablation.py --env metaworld_door-open-v2 --metric success_rate "$@"
