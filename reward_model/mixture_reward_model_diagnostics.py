"""TTP mixture reward model that logs RMS ΔR and SA buffer moments.

Subclasses the production ``mixture_reward_model_alpha_sum_log_over`` model
without modifying it. Diagnostics are written to the existing reward logger
once per ``train_reward`` / ``train_soft_reward`` call (before the update).
"""

from __future__ import annotations

from .diagnostics import (
    compute_reward_buffer_diagnostics,
    log_reward_buffer_diagnostics,
)
from .mixture_reward_model_alpha_sum_log_over import MixtureRewardModel as _BaseMixtureRewardModel

device = "cuda"


class MixtureRewardModel(_BaseMixtureRewardModel):
    def _log_buffer_diagnostics(self):
        if self.logger is None:
            return
        step = int(getattr(self, "total_epochs", 0))
        stats = compute_reward_buffer_diagnostics(
            ensemble=self.ensemble,
            reward_models=self.reward_models,
            ds=self.ds,
            da=self.da,
            device=device,
        )
        log_reward_buffer_diagnostics(self.logger, stats, step)

    def train_reward(self):
        self._log_buffer_diagnostics()
        return super().train_reward()

    def train_soft_reward(self):
        self._log_buffer_diagnostics()
        return super().train_soft_reward()
