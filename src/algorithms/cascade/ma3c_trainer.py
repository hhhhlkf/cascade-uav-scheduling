from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np

from src.algorithms.base_scheduler import BaseScheduler
from src.algorithms.cascade.actor_network import build_actor
from src.algorithms.cascade.critic_network import build_critic
from src.algorithms.cascade.hungarian_match import masked_assignment
from src.algorithms.cascade.state_encoder import CASCADEStateEncoder


@dataclass
class MA3CConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    n_steps: int = 128
    max_grad_norm: float = 0.5
    token_dim: int = 64
    hidden_dim: int = 128
    device: str | None = None


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    next_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute generalized advantage estimates and returns."""
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    next_val = float(next_value)
    for step in reversed(range(len(rewards))):
        non_terminal = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_val * non_terminal - values[step]
        gae = delta + gamma * gae_lambda * non_terminal * gae
        advantages[step] = gae
        next_val = values[step]
    returns = advantages + values
    return advantages, returns


class CASCADEMA3CScheduler(BaseScheduler):
    """CASCADE mA3C+MHSA scheduler with shared UAV actor and command critic."""

    def __init__(self, max_ready_tasks: int, num_uavs: int, config: MA3CConfig | None = None):
        super().__init__(max_ready_tasks, num_uavs)
        self.config = config or MA3CConfig()
        self.encoder = CASCADEStateEncoder(
            max_ready_tasks=max_ready_tasks,
            num_uavs=num_uavs,
            max_nodes=num_uavs + 1,
            max_tasks_total=max_ready_tasks,
            token_dim=self.config.token_dim,
            device=self.config.device,
        )
        self.torch = self.encoder.torch
        self.actor = build_actor(
            self.encoder.global_dim + self.encoder.local_dim,
            self.config.hidden_dim,
            max_ready_tasks,
        ).to(self.encoder.device)
        self.critic = build_critic(self.encoder.global_dim, self.config.hidden_dim).to(self.encoder.device)
        self.actor_optimizer = self.torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.actor.parameters()),
            lr=self.config.actor_lr,
        )
        self.critic_optimizer = self.torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.critic.parameters()),
            lr=self.config.critic_lr,
        )

    def decide(self, action_mask: np.ndarray) -> np.ndarray:
        action = self._empty_action(action_mask)
        if self.last_obs is None:
            return action
        self.encoder.module.eval()
        self.actor.eval()
        with self.torch.no_grad():
            global_state, local_obs = self.encoder.encode(self.last_obs)
            logits = self._actor_logits(global_state, local_obs)[0].transpose(0, 1)
            scores = self._masked_task_probabilities(logits, action_mask)
        for task_idx, uav_idx in masked_assignment(scores, action_mask):
            action[task_idx, uav_idx] = scores[task_idx, uav_idx]
        return action

    def learn(self, batch: Dict) -> Dict[str, float]:
        if not {"rewards", "values", "dones"}.issubset(batch):
            return {"loss_actor": 0.0, "loss_critic": 0.0, "entropy": 0.0}
        advantages, returns = compute_gae(
            batch["rewards"],
            batch["values"],
            batch["dones"],
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            next_value=float(batch.get("next_value", 0.0)),
        )
        return {
            "advantage_mean": float(np.mean(advantages)) if advantages.size else 0.0,
            "return_mean": float(np.mean(returns)) if returns.size else 0.0,
            "loss_actor": 0.0,
            "loss_critic": 0.0,
            "entropy": 0.0,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save(
            {
                "config": self.config.__dict__,
                "encoder": self.encoder.state_dict(),
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        checkpoint = self.torch.load(Path(path), map_location=self.encoder.device)
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])

    def value(self, obs: Dict[str, np.ndarray]) -> float:
        self.encoder.module.eval()
        self.critic.eval()
        with self.torch.no_grad():
            global_state, _ = self.encoder.encode(obs)
            return float(self.critic(global_state).squeeze().detach().cpu().item())

    def _actor_logits(self, global_state, local_obs):
        batch, num_uavs, _ = local_obs.shape
        global_per_uav = global_state.unsqueeze(1).expand(batch, num_uavs, global_state.shape[-1])
        actor_input = self.torch.cat([global_per_uav, local_obs], dim=-1)
        return self.actor(actor_input.reshape(batch * num_uavs, -1)).reshape(batch, num_uavs, self.max_ready_tasks)

    def _masked_task_probabilities(self, logits, action_mask: np.ndarray) -> np.ndarray:
        mask = self.torch.as_tensor(action_mask, dtype=self.torch.bool, device=logits.device)
        masked_logits = logits.masked_fill(~mask, -1e9)
        probs = self.torch.softmax(masked_logits, dim=0).masked_fill(~mask, 0.0)
        empty_cols = ~mask.any(dim=0)
        if empty_cols.any():
            probs[:, empty_cols] = 0.0
        return probs.detach().cpu().numpy().astype(np.float32)
