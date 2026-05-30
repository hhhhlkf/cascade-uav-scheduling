from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

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
        self.optimizer = self.torch.optim.Adam(
            [
                {"params": self.encoder.parameters(), "lr": self.config.actor_lr},
                {"params": self.actor.parameters(), "lr": self.config.actor_lr},
                {"params": self.critic.parameters(), "lr": self.config.critic_lr},
            ]
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
            padded_mask = self._pad_action_mask(action_mask)
            scores = self._masked_task_probabilities(logits, padded_mask)[: action_mask.shape[0], : action_mask.shape[1]]
        for task_idx, uav_idx in masked_assignment(scores, action_mask):
            action[task_idx, uav_idx] = scores[task_idx, uav_idx]
        return action

    def decide_with_trace(self, obs: Dict[str, np.ndarray], action_mask: np.ndarray) -> tuple[np.ndarray, Dict]:
        self.encoder.module.train()
        self.actor.train()
        self.critic.train()
        global_state, local_obs = self.encoder.encode_train(obs)
        logits = self._actor_logits(global_state, local_obs)[0].transpose(0, 1)
        value = self.critic(global_state).squeeze()
        padded_mask = self._pad_action_mask(action_mask)
        probs = self._masked_task_probabilities_tensor(logits, padded_mask)
        scores = probs[: action_mask.shape[0], : action_mask.shape[1]].detach().cpu().numpy().astype(np.float32)
        action = self._empty_action(action_mask)
        assignments, log_prob, entropy = self._sample_assignments(probs, action_mask)
        for task_idx, uav_idx in assignments:
            action[task_idx, uav_idx] = scores[task_idx, uav_idx]
        return action, {"log_prob": log_prob, "entropy": entropy, "value": value}

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

    def learn_episode(self, traces: List[Dict], rewards: List[float], dones: List[bool], next_value: float = 0.0) -> Dict[str, float]:
        if not traces:
            return {"loss_actor": 0.0, "loss_critic": 0.0, "entropy": 0.0}
        values_np = np.asarray([float(trace["value"].detach().cpu().item()) for trace in traces], dtype=np.float32)
        advantages_np, returns_np = compute_gae(
            np.asarray(rewards, dtype=np.float32),
            values_np,
            np.asarray(dones, dtype=np.float32),
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            next_value=next_value,
        )
        advantages = self.torch.as_tensor(advantages_np, dtype=self.torch.float32, device=self.encoder.device)
        returns = self.torch.as_tensor(returns_np, dtype=self.torch.float32, device=self.encoder.device)
        log_probs = self.torch.stack([trace["log_prob"] for trace in traces])
        values = self.torch.stack([trace["value"] for trace in traces]).reshape(-1)
        entropies = self.torch.stack([trace["entropy"] for trace in traces])
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = self.torch.nn.functional.mse_loss(values, returns)
        entropy = entropies.mean()
        loss = actor_loss + 0.5 * critic_loss - self.config.entropy_coef * entropy
        self.optimizer.zero_grad()
        loss.backward()
        params = list(self.encoder.parameters()) + list(self.actor.parameters()) + list(self.critic.parameters())
        self.torch.nn.utils.clip_grad_norm_(params, self.config.max_grad_norm)
        self.optimizer.step()
        return {
            "loss_actor": float(actor_loss.detach().cpu().item()),
            "loss_critic": float(critic_loss.detach().cpu().item()),
            "entropy": float(entropy.detach().cpu().item()),
            "advantage_mean": float(np.mean(advantages_np)) if advantages_np.size else 0.0,
            "return_mean": float(np.mean(returns_np)) if returns_np.size else 0.0,
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
        return self._masked_task_probabilities_tensor(logits, action_mask).detach().cpu().numpy().astype(np.float32)

    def _masked_task_probabilities_tensor(self, logits, action_mask: np.ndarray):
        mask = self.torch.as_tensor(action_mask, dtype=self.torch.bool, device=logits.device)
        masked_logits = logits.masked_fill(~mask, -1e9)
        probs = self.torch.softmax(masked_logits, dim=0).masked_fill(~mask, 0.0)
        empty_cols = ~mask.any(dim=0)
        if empty_cols.any():
            probs[:, empty_cols] = 0.0
        return probs

    def _pad_action_mask(self, action_mask: np.ndarray) -> np.ndarray:
        padded = np.zeros((self.max_ready_tasks, self.num_uavs), dtype=np.float32)
        rows = min(action_mask.shape[0], self.max_ready_tasks)
        cols = min(action_mask.shape[1], self.num_uavs)
        padded[:rows, :cols] = action_mask[:rows, :cols]
        return padded

    def _assignment_log_prob(self, probs, assignments: list[tuple[int, int]]):
        if not assignments:
            return probs.sum() * 0.0
        selected = self.torch.stack([probs[task_idx, uav_idx].clamp_min(1e-8).log() for task_idx, uav_idx in assignments])
        return selected.sum()

    def _sample_assignments(self, probs, action_mask: np.ndarray):
        """Sample a one-to-one assignment for policy-gradient training.

        Evaluation still uses deterministic Hungarian decoding in decide(). During
        training, actions must be drawn from the policy distribution so the
        collected log_prob matches the action generation process.
        """
        rows, cols = action_mask.shape
        mask = self.torch.as_tensor(action_mask[:rows, :cols] > 0.0, dtype=self.torch.bool, device=probs.device)
        used_tasks = self.torch.zeros(rows, dtype=self.torch.bool, device=probs.device)
        assignments: list[tuple[int, int]] = []
        log_probs = []
        entropies = []
        for uav_idx_tensor in self.torch.randperm(cols, device=probs.device):
            uav_idx = int(uav_idx_tensor.detach().cpu().item())
            valid_tasks = mask[:, uav_idx] & ~used_tasks
            if not bool(valid_tasks.any()):
                continue
            task_probs = probs[:rows, uav_idx][valid_tasks]
            prob_sum = task_probs.sum()
            if float(prob_sum.detach().cpu().item()) <= 0.0:
                continue
            task_probs = task_probs / prob_sum.clamp_min(1e-8)
            dist = self.torch.distributions.Categorical(probs=task_probs)
            sampled_local_idx = dist.sample()
            valid_indices = self.torch.nonzero(valid_tasks, as_tuple=False).reshape(-1)
            task_idx = int(valid_indices[sampled_local_idx].detach().cpu().item())
            assignments.append((task_idx, uav_idx))
            log_probs.append(dist.log_prob(sampled_local_idx))
            entropies.append(dist.entropy())
            used_tasks[task_idx] = True
        if not log_probs:
            zero = probs.sum() * 0.0
            return [], zero, zero
        return assignments, self.torch.stack(log_probs).sum(), self.torch.stack(entropies).mean()

    def _masked_entropy(self, probs):
        valid = probs > 0.0
        if not valid.any():
            return probs.sum() * 0.0
        return -(probs[valid] * probs[valid].clamp_min(1e-8).log()).mean()


def build_cascade_scheduler(
    max_ready_tasks: int,
    num_uavs: int = 15,
    config: MA3CConfig | None = None,
    checkpoint: str | Path | None = None,
) -> CASCADEMA3CScheduler:
    scheduler = CASCADEMA3CScheduler(max_ready_tasks, num_uavs, config=config)
    if checkpoint:
        scheduler.load(checkpoint)
    return scheduler


def cascade_factory(
    max_ready_tasks: int,
    model_num_uavs: int = 15,
    config: MA3CConfig | None = None,
    checkpoint: str | Path | None = None,
):
    def _factory(_env_max_ready_tasks: int, _env_num_uavs: int) -> CASCADEMA3CScheduler:
        return build_cascade_scheduler(
            max_ready_tasks=max(max_ready_tasks, _env_max_ready_tasks),
            num_uavs=max(model_num_uavs, _env_num_uavs),
            config=config,
            checkpoint=checkpoint,
        )

    return _factory
