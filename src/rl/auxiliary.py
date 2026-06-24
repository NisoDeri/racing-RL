"""Auxiliary prediction heads for Phase 7 PPO experiments."""

from __future__ import annotations

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy


class AuxRaycastActorCriticPolicy(ActorCriticPolicy):
    """Actor-critic policy with a head that predicts the next raycast frame."""

    def __init__(self, *args, auxiliary_raycast_dim=30, **kwargs):
        self.auxiliary_raycast_dim = int(auxiliary_raycast_dim)
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        action_dim = int(np.prod(self.action_space.shape))
        latent_dim = int(self.mlp_extractor.latent_dim_pi)
        hidden_dim = max(latent_dim, self.auxiliary_raycast_dim)
        self.auxiliary_head = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.auxiliary_raycast_dim),
            nn.Sigmoid(),
        ).to(self.device)
        self._init_auxiliary_head()

        # The parent class builds the optimizer before this head exists.
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _init_auxiliary_head(self):
        linear_layers = [
            module for module in self.auxiliary_head.modules()
            if isinstance(module, nn.Linear)
        ]
        for index, layer in enumerate(linear_layers):
            gain = 0.01 if index == len(linear_layers) - 1 else math.sqrt(2)
            nn.init.orthogonal_(layer.weight, gain=gain)
            nn.init.constant_(layer.bias, 0.0)

    def predict_next_rays(self, obs, actions):
        features = self.extract_features(obs)
        if isinstance(features, tuple):
            features = features[0]
        latent_pi = self.mlp_extractor.forward_actor(features)
        actions = actions.float().reshape(actions.shape[0], -1)
        return self.auxiliary_head(th.cat([latent_pi, actions], dim=1))


class AuxRaycastPredictionCallback(BaseCallback):
    """Train the auxiliary next-raycast head from consecutive rollout frames."""

    def __init__(
        self,
        loss_coef=0.05,
        raycast_dim=30,
        frame_dim=34,
        batch_size=256,
        gradient_steps=1,
        verbose=0,
    ):
        super().__init__(verbose=verbose)
        self.loss_coef = float(loss_coef)
        self.raycast_dim = int(raycast_dim)
        self.frame_dim = int(frame_dim)
        self.batch_size = int(batch_size)
        self.gradient_steps = int(gradient_steps)

    def _on_training_start(self):
        if not hasattr(self.model.policy, "predict_next_rays"):
            raise TypeError(
                "AuxRaycastPredictionCallback requires AuxRaycastActorCriticPolicy"
            )

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        if self.loss_coef == 0.0:
            return

        obs, actions, target_rays = self._build_auxiliary_dataset()
        if obs is None:
            return

        losses = []
        n_samples = obs.shape[0]
        for _ in range(self.gradient_steps):
            indices = np.random.permutation(n_samples)
            for start in range(0, n_samples, self.batch_size):
                batch_indices = indices[start:start + self.batch_size]
                obs_batch = th.as_tensor(
                    obs[batch_indices], device=self.model.device
                ).float()
                action_batch = th.as_tensor(
                    actions[batch_indices], device=self.model.device
                ).float()
                target_batch = th.as_tensor(
                    target_rays[batch_indices], device=self.model.device
                ).float()

                prediction = self.model.policy.predict_next_rays(
                    obs_batch, action_batch
                )
                loss = F.mse_loss(prediction, target_batch) * self.loss_coef

                self.model.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(
                    self.model.policy.parameters(), self.model.max_grad_norm
                )
                self.model.policy.optimizer.step()
                losses.append(float(loss.detach().cpu().item()))

        if losses:
            self.logger.record("train/aux_raycast_loss", float(np.mean(losses)))

    def _build_auxiliary_dataset(self):
        rollout_buffer = self.model.rollout_buffer
        if rollout_buffer.buffer_size < 2:
            return None, None, None

        observations = rollout_buffer.observations
        actions = rollout_buffer.actions
        episode_starts = rollout_buffer.episode_starts
        if isinstance(observations, dict):
            raise TypeError("Auxiliary raycast prediction expects Box observations")

        valid = episode_starts[1:] == 0.0
        if not np.any(valid):
            return None, None, None

        obs = observations[:-1][valid]
        next_obs = observations[1:][valid]
        actions = actions[:-1][valid]

        obs_dim = int(next_obs.shape[-1])
        frame_start = obs_dim - self.frame_dim
        if frame_start < 0 or frame_start + self.raycast_dim > obs_dim:
            raise ValueError(
                "Cannot locate raycast targets in rollout observations: "
                f"obs_dim={obs_dim}, frame_dim={self.frame_dim}, "
                f"raycast_dim={self.raycast_dim}"
            )
        target_rays = next_obs[..., frame_start:frame_start + self.raycast_dim]
        return obs, actions, target_rays
