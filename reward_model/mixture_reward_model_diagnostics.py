"""TTP mixture reward model with an explicit buffer-diagnostics hook.

Subclasses the production ``mixture_reward_model_alpha_sum_log_over`` model
without modifying it. Call ``log_buffer_diagnostics(step)`` when you want a
snapshot (e.g. once after ``num_seed_steps + num_unsup_steps``). Training
itself is unchanged and does not auto-log diagnostics.
"""

from __future__ import annotations

from .diagnostics import (
    compute_reward_buffer_diagnostics,
    log_reward_buffer_diagnostics,
)
from .mixture_reward_model_alpha_sum_log_over import MixtureRewardModel as _BaseMixtureRewardModel

device = "cuda"


class MixtureRewardModel(_BaseMixtureRewardModel):
    def log_buffer_diagnostics(self, step: int):
        """Log rms|ΔR|_0 and SA moments, then flush the reward CSV row."""
        if self.logger is None:
            return
        stats = compute_reward_buffer_diagnostics(
            ensemble=self.ensemble,
            reward_models=self.reward_models,
            ds=self.ds,
            da=self.da,
            device=device,
        )
        log_reward_buffer_diagnostics(self.logger, stats, step)
        self.logger.dump(step, ty="reward")
        print(
            f"[diagnostics @ step={step}] "
            f"rms_delta_r={stats['rms_delta_r']:.6f} "
            f"mean_sa_var={stats['mean_sa_var']:.6f} "
            f"mean_sa_second_moment={stats['mean_sa_second_moment']:.6f} "
            f"n_pairs={int(stats['n_pairs'])} "
            f"n_transitions={int(stats['n_transitions'])}"
        )
