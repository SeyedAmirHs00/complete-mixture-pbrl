"""TTP mixture with two-path w_k loss, Adam on the reward net, SGD on α.

Same loss as ``mixture_reward_model_wk_sgd.py`` / ``fig6_alpha_curve_train_wk.py``:

    trust = tanh(α)
    coef  = trust / max|trust|          # max detached
    w_k   = K |trust| / Σ|trust|        # detached; reward path only

    loss_R = mean(w_k * CE(coef.detach() * logits, y))   # updates reward net (Adam)
    loss_A = mean(CE(coef * logits.detach(), y))         # updates α (SGD)

Default LRs: reward Adam ``0.0003``, alpha SGD ``0.005``.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np

from .vanilla_reward_model import gen_net
from .mixture_reward_model_wk_sgd import MixtureBufferDataset, detached_wk
from .mixture_reward_model_wk_sgd import MixtureRewardModel as _WkSgdMixtureRewardModel

from typing import List, Union
from utils.logger import Logger
from .vanilla_reward_model import RewardModel


device = "cuda"


class MixtureRewardModel(_WkSgdMixtureRewardModel):
    def __init__(
        self,
        reward_models: Union[List[RewardModel], None],
        ds,
        da,
        ensemble_size=3,
        mb_size=128,
        lr=0.0003,
        alpha_lr=0.005,
        size_segment=1,
        env_maker=None,
        max_size=100,
        activation="tanh",
        capacity=5e5,
        large_batch=1,
        label_margin=0.0,
        logger: Union[Logger, None] = None,
        entropy_coef=0.05,
        init_trust=0.01,
    ):
        super().__init__(
            reward_models,
            ds,
            da,
            ensemble_size=ensemble_size,
            mb_size=mb_size,
            lr=lr,
            alpha_lr=alpha_lr,
            size_segment=size_segment,
            env_maker=env_maker,
            max_size=max_size,
            activation=activation,
            capacity=capacity,
            large_batch=large_batch,
            label_margin=label_margin,
            logger=logger,
            entropy_coef=entropy_coef,
            init_trust=init_trust,
        )

    def construct_ensemble(self):
        for i in range(self.de):
            model = nn.Sequential(
                *gen_net(
                    in_size=self.ds + self.da,
                    out_size=1,
                    H=256,
                    n_layers=3,
                    activation=self.activation,
                )
            ).float().to(device)
            self.ensemble.append(model)
            self.paramlst.extend(model.parameters())

        alphas_tensor = self.init_trust * torch.ones(
            len(self.reward_models), dtype=torch.float32, device=device
        )
        self.alphas = nn.Parameter(alphas_tensor)
        self.paramlst.append(self.alphas)
        net_params = [p for p in self.paramlst if p is not self.alphas]
        self.opt_reward = torch.optim.Adam(net_params, lr=self.lr)
        self.opt_alpha = torch.optim.SGD([self.alphas], lr=self.alpha_lr)
        # Keep ``self.opt`` for any base-class code that expects it (save/load).
        self.opt = self.opt_reward
        print(
            f"reward optimizer: Adam  network_lr={self.lr}  |  "
            f"alpha optimizer: SGD  alpha_lr={self.alpha_lr}"
        )

    def _train_reward_common(self, use_soft_loss=False):
        dataset = MixtureBufferDataset(self.reward_models)
        dataloaders = [
            DataLoader(
                dataset,
                batch_size=self.train_batch_size,
                shuffle=True,
                num_workers=2,
            )
            for _ in range(self.de)
        ]
        loaders = [iter(dl) for dl in dataloaders]

        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.zeros(self.de, dtype=np.int64)
        total = np.zeros(self.de, dtype=np.int64)
        k = len(self.reward_models)
        net_params = [p for p in self.paramlst if p is not self.alphas]

        while True:
            self.opt_reward.zero_grad(set_to_none=True)
            self.opt_alpha.zero_grad(set_to_none=True)
            loss = 0.0

            is_finished = False
            for m, loader in enumerate(loaders):
                try:
                    seg1, seg2, labels, expert_inds, expert_data_counters = next(loader)
                    batch_size = labels.size(0)
                    total[m] += batch_size
                except StopIteration:
                    is_finished = True
                    break

                seg1 = seg1.to(device)
                seg2 = seg2.to(device)
                labels = labels.to(device)
                expert_inds = expert_inds.to(device)

                r1 = self.ensemble[m](torch.cat((seg1,), dim=1)).sum(dim=1)
                r2 = self.ensemble[m](torch.cat((seg2,), dim=1)).sum(dim=1)
                delta_logits = torch.stack([r1, r2], dim=1)

                trust = torch.tanh(self.alphas)
                coef = trust / trust.abs().amax().clamp_min(1e-12).detach()
                w = detached_wk(trust, k)
                coef_m = coef[expert_inds]
                w_m = w[expert_inds]

                logits_R = coef_m.detach().view(-1, 1, 1) * delta_logits
                ce_R, labels_mod = self._pref_loss(logits_R, labels, use_soft_loss)
                w_view = w_m.reshape(ce_R.shape[0], *([1] * (ce_R.dim() - 1)))
                loss_R = (w_view * ce_R).mean()

                logits_A = coef_m.view(-1, 1, 1) * delta_logits.detach()
                ce_A, _ = self._pref_loss(logits_A, labels, use_soft_loss)
                loss_A = ce_A.mean()

                cur_loss = loss_R + loss_A
                loss = loss + cur_loss
                ensemble_losses[m].append(cur_loss.item())

                _, preds = logits_R.max(dim=1)
                ensemble_acc[m] += (preds == labels_mod).sum().item()

            if is_finished:
                break

            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_params, 10.0)
            self.opt_reward.step()
            self.opt_alpha.step()

            self.total_epochs += 1
            with torch.no_grad():
                alphas_tan = torch.tanh(self.alphas)
                alphas_tan_abs = alphas_tan.abs()
                logits_coef = alphas_tan / alphas_tan_abs.max().clamp_min(1e-8)
                w_log = detached_wk(alphas_tan, k)
                if self.logger is not None:
                    self.logger.log(
                        "reward/alpha_abs_sum",
                        self.alphas.abs().sum().item(),
                        self.total_epochs,
                    )
                    for i, coef_i in enumerate(logits_coef):
                        self.logger.log(
                            f"reward/expert_logits_coef_{i}",
                            coef_i.item(),
                            self.total_epochs,
                        )
                    for i, w_i in enumerate(w_log):
                        self.logger.log(
                            f"reward/expert_coef_{i}", w_i.item(), self.total_epochs
                        )
                    for i, alpha in enumerate(self.alphas):
                        self.logger.log(
                            f"reward/alpha_{i}", alpha.item(), self.total_epochs
                        )
                    for i, alpha in enumerate(alphas_tan):
                        self.logger.log(
                            f"reward/alpha_tan_{i}", alpha.item(), self.total_epochs
                        )

        if self.logger is not None:
            self.logger.dump(self.total_epochs, ty="reward")
        ensemble_acc = ensemble_acc / total
        return ensemble_acc
