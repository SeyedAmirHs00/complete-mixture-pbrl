import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import math
import torch.nn.functional as F
import torch.optim as optim
# from torch.optim.lr_scheduler import ExponentialLR
from utils.logger import Logger  

import numpy as np

from .vanilla_reward_model import RewardModel
from .vanilla_reward_model import gen_net
from .vanilla_reward_model import RATIONAL_TEACHER

from typing import List, Union
import os

NEAR_ONE = 1 - 1e-6

device = "cuda"

def torch_tanh(x):
    # Works for tensors and scalars, assumes |x| < 1
    return torch.tanh(x)

class MixtureBufferDataset(Dataset):
    def __init__(self, reward_models):
        """
        reward_models: list of RewardModel, each with buffer_seg1, buffer_seg2, buffer_label
        """
        self.examples = []
        self.expert_data_counter = [0 for _ in range(len(reward_models))]
        self.total_data_counter = 0
        for expert_idx, rm in enumerate(reward_models):
            N = len(rm.buffer_label) if rm.buffer_full else rm.buffer_index
            self.expert_data_counter[expert_idx] = N
            self.total_data_counter += N
            for i in range(N):
                seg1 = rm.buffer_seg1[i]
                seg2 = rm.buffer_seg2[i]
                label = rm.buffer_label[i]
                self.examples.append((seg1, seg2, label, expert_idx))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        seg1, seg2, label, expert_idx = self.examples[idx]
        return (torch.from_numpy(seg1).float(),
                torch.from_numpy(seg2).float(),
                torch.tensor(label, dtype=torch.long),
                torch.tensor(expert_idx, dtype=torch.long),
                torch.tensor(self.expert_data_counter[expert_idx], dtype=torch.float))


class MixtureRewardModel(RewardModel):
    def __init__(self, reward_models: Union[List[RewardModel], None], ds, da, l2_factor=0.1,
                 ensemble_size=3, mb_size=128, lr=3e-4, size_segment=1,
                 env_maker=None, max_size=100, activation='tanh',capacity=5e5,
                 large_batch=1, label_margin=0.0, logger: Union[Logger, None] = None, entropy_coef=0.05, init_trust=0.01):
        if reward_models is None:
            reward_models = [
                            RewardModel(ds, da, ensemble_size=ensemble_size, 
                                         lr=lr, mb_size=mb_size, size_segment=size_segment,
                                         env_maker=env_maker, max_size=max_size,
                                         activation=activation, capacity=capacity, large_batch=large_batch,
                                         label_margin=label_margin,
                                         teacher_beta=1, teacher_gamma=1),
                            RewardModel(ds, da, ensemble_size=ensemble_size, 
                                         lr=lr, mb_size=mb_size, size_segment=size_segment,
                                         env_maker=env_maker, max_size=max_size,
                                         activation=activation, capacity=capacity, large_batch=large_batch,
                                         label_margin=label_margin,
                                         teacher_beta=1, teacher_gamma=1),
                            RewardModel(ds, da, ensemble_size=ensemble_size, 
                                         lr=lr, mb_size=mb_size, size_segment=size_segment,
                                         env_maker=env_maker, max_size=max_size,
                                         activation=activation, capacity=capacity, large_batch=large_batch,
                                         label_margin=label_margin,
                                         teacher_beta=1, teacher_gamma=1),
                            RewardModel(ds, da, ensemble_size=ensemble_size, 
                                         lr=lr, mb_size=mb_size, size_segment=size_segment,
                                         env_maker=env_maker, max_size=max_size,
                                         activation=activation, capacity=capacity, large_batch=large_batch,
                                         label_margin=label_margin,
                                         teacher_beta=-1, teacher_gamma=1),
                            ]
        self.ds = ds
        self.da = da
        self.reward_models = reward_models
        self.de = ensemble_size
        self.mb_size = mb_size
        self.origin_mb_size = mb_size
        self.lr = lr
        self.size_segment = size_segment
        self.env_maker = env_maker
        self.max_size = max_size
        self.capacity = capacity
        self.large_batch = large_batch
        self.label_margin = label_margin
        self.l2_factor = 0
        # self.l2_factor = l2_factor
        self.entropy_coef = entropy_coef

        self.ensemble = []
        self.alphas = []
        self.paramlst = []
        self.init_trust = init_trust
        self.opt = None
        self.activation = activation
        
        self.construct_ensemble()
        self.CEloss = nn.CrossEntropyLoss(reduction="none")
        self.train_batch_size = 128

        self.label_margin = label_margin
        self.label_target = 1 - 2*self.label_margin

        self.total_epochs = 0

        self.logger = logger

        #dummy variables
        self.running_means = []
        self.running_stds = []
        self.best_seg = []
        self.best_label = []
        self.best_action = []
        self.inputs = []
        self.targets = []
        self.raw_actions = []
        self.img_inputs = []
        self.teacher_beta = -1
        self.teacher_gamma = 1
        self.teacher_eps_mistake = 0
        self.teacher_eps_equal = 0
        self.teacher_eps_skip = 0
        self.teacher_thres_skip = 0
        self.teacher_thres_equal = 0

        # Check equality of size_segments across all reward models
        for reward_model in self.reward_models:
            if reward_model.size_segment != size_segment:
                raise ValueError(f"Size segment mismatch: {reward_model.size_segment} != {size_segment}")

        # Check equality of ensemble_size across all reward models
        for reward_model in self.reward_models:
            if reward_model.de != ensemble_size:
                raise ValueError(f"Ensemble size mismatch: {reward_model.de} != {ensemble_size}")
        
        # Check equality of max_size across all reward models
        for reward_model in self.reward_models:
            if reward_model.max_size != max_size:
                raise ValueError(f"Max size mismatch: {reward_model.max_size} != {max_size}")
        
        # Check equality of size_segment across all reward models
        for reward_model in self.reward_models:
            if reward_model.size_segment != size_segment:
                raise ValueError(f"Size segment mismatch: {reward_model.size_segment} != {size_segment}")
        
        # Check equality of capacity across all reward models
        for reward_model in self.reward_models:
            if reward_model.capacity != capacity:
                raise ValueError(f"Capacity mismatch: {reward_model.capacity} != {capacity}")
        
        # Check equality of label_margin across all reward models
        for reward_model in self.reward_models:
            if reward_model.label_margin != label_margin:
                raise ValueError(f"Label margin mismatch: {reward_model.label_margin} != {label_margin}")

        # Check equality of ensemble_size across all reward models
        for reward_model in self.reward_models:
            if reward_model.de != self.de:
                raise ValueError(f"Ensemble size mismatch: {reward_model.de} != {self.de}")

        for reward_model in self.reward_models:
            reward_model.ensemble = self.ensemble
        
    def get_entropy_loss(self, x_1, x_2):
        # get probability x_1 > x_2
        probs = []
        for member in range(self.de):
            probs.append(self.p_hat_entropy_loss(x_1, x_2, member=member))
        probs_t = torch.stack(probs)
        return torch.mean(probs_t, axis=0), torch.std(probs_t, axis=0)

    def p_hat_member_loss(self, x_1, x_2, member=-1):
        # softmaxing to get the probabilities according to eqn 1
        r_hat1 = self.r_hat_member(x_1, member=member)
        r_hat2 = self.r_hat_member(x_2, member=member)
        r_hat1 = r_hat1.sum(axis=1)
        r_hat2 = r_hat2.sum(axis=1)
        r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
        
        # taking 0 index for probability x_1 > x_2
        return F.softmax(r_hat, dim=-1)[:,0]
    
    def p_hat_entropy_loss(self, x_1, x_2, member=-1):
        # softmaxing to get the probabilities according to eqn 1
        r_hat1 = self.r_hat_member(x_1, member=member)
        r_hat2 = self.r_hat_member(x_2, member=member)
        r_hat1 = r_hat1.sum(axis=1)
        r_hat2 = r_hat2.sum(axis=1)
        r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
        
        ent = F.softmax(r_hat, dim=-1) * F.log_softmax(r_hat, dim=-1)
        ent = ent.sum(axis=-1).abs()
        return ent

    def softXEnt_loss(self, input, target):
        logprobs = torch.nn.functional.log_softmax (input, dim = 1)
        return  -(target * logprobs).sum() / input.shape[0]
    

    def change_batch(self, new_frac):
        self.mb_size = int(self.origin_mb_size*new_frac)
        for reward_model in self.reward_models:
            reward_model.change_batch(new_frac)
    
    def set_batch(self, new_batch):
        self.mb_size = int(new_batch)
        for reward_model in self.reward_models:
            reward_model.set_batch(new_batch)
        
    def set_teacher_thres_skip(self, new_margin):
        self.teacher_thres_skip = new_margin * self.teacher_eps_skip
        
    def set_teacher_thres_equal(self, new_margin):
        self.teacher_thres_equal = new_margin * self.teacher_eps_equal
                
    def construct_ensemble(self):
        for _ in range(self.de):
            model = nn.Sequential(*gen_net(in_size=self.ds+self.da, out_size=1,
                                            H=256, n_layers=3, 
                                            activation=self.activation)
                                            ).float().to(device)
            self.ensemble.append(model)
            self.paramlst.extend(model.parameters())
        alphas_tensor = self.init_trust * torch.ones(len(self.reward_models), dtype=torch.float32, device=device)
        self.alphas = nn.Parameter(alphas_tensor)
        self.paramlst.append(self.alphas)
        self.opt = optim.Adam(self.paramlst, lr=self.lr)
        # self.sched = ExponentialLR(self.opt, gamma=0.999999)
        
    def add_data(self, obs, act, rew, done):
        for reward_model in self.reward_models:
            reward_model.add_data(obs, act, rew, done)

    def add_data_batch(self, obses, rewards):
        for reward_model in self.reward_models:
            reward_model.add_data_batch(obses, rewards)
    
    def r_hat_member(self, x, member=-1):
        return self.ensemble[member](torch.from_numpy(x).float().to(device))
    
    def r_hat(self, x):
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)
        return np.mean(r_hats)
    
    def r_hat_batch(self, x):
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)
        return np.mean(r_hats, axis=0)

    def save(self, work_dir, step):
        os.makedirs(work_dir, exist_ok=True)
        # Save ensemble models
        for idx, model in enumerate(self.ensemble):
            torch.save(model.state_dict(), os.path.join(work_dir, f'ensemble_{idx}_step_{step}.pt'))
        # Save alphas
        torch.save(self.alphas.data, os.path.join(work_dir, f'alphas_step_{step}.pt'))
        # Save reward models
        for idx, reward_model in enumerate(self.reward_models):
            if hasattr(reward_model, 'save'):
                reward_model.save(work_dir, f'{step}_reward_model_{idx}')
            else:
                torch.save(reward_model.state_dict(), os.path.join(work_dir, f'reward_model_{idx}_step_{step}.pt'))

    @classmethod
    def load(cls, work_dir, step, ds, da, reward_models=None, **kwargs):
        # Instantiate the object
        obj = cls(reward_models, ds, da, **kwargs)
        # Load ensemble models
        for idx, model in enumerate(obj.ensemble):
            model_path = os.path.join(work_dir, f'ensemble_{idx}_step_{step}.pt')
            model.load_state_dict(torch.load(model_path, map_location=device))
        # Load alphas
        alphas_path = os.path.join(work_dir, f'alphas_step_{step}.pt')
        obj.alphas.data = torch.load(alphas_path, map_location=device)
        # Load reward models
        for idx, reward_model in enumerate(obj.reward_models):
            if hasattr(reward_model, 'load'):
                obj.reward_models[idx] = reward_model.load(work_dir, f'{step}_reward_model_{idx}', ds, da)
            else:
                model_path = os.path.join(work_dir, f'reward_model_{idx}_step_{step}.pt')
                reward_model.load_state_dict(torch.load(model_path, map_location=device))
        return obj

    def uniform_sampling(self):
        cnt_labels = [reward_model.uniform_sampling() for reward_model in self.reward_models]
        return sum(cnt_labels)
    
    def disagreement_sampling(self):
        cnt_labels = [reward_model.disagreement_sampling() for reward_model in self.reward_models]
        return sum(cnt_labels)
    
    def entropy_sampling(self):
        cnt_labels = [reward_model.entropy_sampling() for reward_model in self.reward_models]
        return sum(cnt_labels)
    
    def kcenter_sampling(self):
        cnt_labels = [reward_model.kcenter_sampling() for reward_model in self.reward_models]
        return sum(cnt_labels)
    
    def kcenter_disagree_sampling(self):
        cnt_labels = [reward_model.kcenter_disagree_sampling() for reward_model in self.reward_models]
        return sum(cnt_labels)
    
    def kcenter_entropy_sampling(self):
        cnt_labels = [reward_model.kcenter_entropy_sampling() for reward_model in self.reward_models]
        return sum(cnt_labels)
    
    def _train_reward_common(self, use_soft_loss=False):
        dataset = MixtureBufferDataset(self.reward_models)
        # Create DataLoaders
        dataloaders = [
            DataLoader(
                dataset,
                batch_size=self.train_batch_size,
                shuffle=True,
                num_workers=2
            )
            for _ in range(self.de)
        ]

        # Convert to iterators once per epoch
        loaders = [iter(dl) for dl in dataloaders]

        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc    = np.zeros(self.de, dtype=np.int64)
        total           = 0

        while True:
            self.opt.zero_grad()
            loss = 0.0

            is_finished = False
            for m, loader in enumerate(loaders):
                try:
                    seg1, seg2, labels, expert_inds, expert_data_counters = next(loader)
                    batch_size = labels.size(0)
                    total += batch_size
                except StopIteration:
                    is_finished = True
                    break

                seg1 = seg1.to(device)
                seg2 = seg2.to(device)
                labels = labels.to(device)
                expert_inds = expert_inds.to(device)
                expert_data_counters = expert_data_counters.to(device)

                r1 = self.ensemble[m](torch.cat((seg1,), dim=1)).sum(dim=1)
                r2 = self.ensemble[m](torch.cat((seg2,), dim=1)).sum(dim=1)
                logits = torch.stack([r1, r2], dim=1)

                alphas_m = self.alphas[expert_inds]
                alphas_m_tan = torch_tanh(alphas_m)
                logits = logits * alphas_m_tan.view(-1, 1, 1)

                if use_soft_loss:
                    uniform_index = labels == -1
                    labels_mod = labels.clone()
                    labels_mod[uniform_index] = 0
                    target_onehot = torch.zeros_like(logits).scatter(
                        1, labels_mod.unsqueeze(1), self.label_target
                    )
                    target_onehot += self.label_margin
                    if sum(uniform_index) > 0:
                        target_onehot[uniform_index] = 0.5
                    cur_loss = self.softXEnt_loss(logits, target_onehot)
                else:
                    cur_loss = self.CEloss(logits, labels)

                # alphas_sum = alphas_m.abs().sum()
                # if alphas_sum == 0:
                #     trust_weight = torch.ones_like(alphas_m) / alphas_m.numel()
                # else:
                #     trust_weight = alphas_m.abs() / alphas_sum
                # data_coverage_penalty = torch.exp(
                #     -expert_data_counters / dataset.total_data_counter
                # )
                # cur_loss = trust_weight * data_coverage_penalty * cur_loss
                cur_loss = cur_loss.mean()

                loss += (cur_loss)
                ensemble_losses[m].append(cur_loss.item())

                _, preds = logits.max(dim=1)
                if use_soft_loss:
                    ensemble_acc[m] += (preds == labels_mod).sum().item()
                else:
                    ensemble_acc[m] += (preds == labels).sum().item()

            if is_finished:
                break

            loss.backward()
            self.opt.step()

            
            self.total_epochs += 1
            if self.logger is not None:
                for i, alpha in enumerate(self.alphas):
                    self.logger.log(f"reward/alpha_{i}", alpha.item(), self.total_epochs)
            alphas_tan = torch_tanh(self.alphas)
            if self.logger is not None:
                for i, alpha in enumerate(alphas_tan):
                    self.logger.log(f"reward/alpha_tan_{i}", alpha.item(), self.total_epochs)
            if self.logger is not None:
                self.logger.log("reward/learning_rate", self.opt.param_groups[0]['lr'], self.total_epochs)

        self.logger.dump(self.total_epochs, ty="reward")
        ensemble_acc = ensemble_acc / total
        return ensemble_acc


    def train_reward(self):
        return self._train_reward_common(use_soft_loss=False)

    def train_soft_reward(self):
        return self._train_reward_common(use_soft_loss=True)
        
    

