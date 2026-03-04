import numpy as np
import torch.nn as nn

from typing import Dict, Union, Tuple
from collections import defaultdict
from offlinerlkit.policy.model_based.mopo import MOPOPolicy
from offlinerlkit.dynamics import BaseDynamics
from offlinerlkit.utils.reward_model import SPLRewardModel


class SSLMOPOPolicy(MOPOPolicy):
    def __init__(
        self,
        dynamics: BaseDynamics,
        reward_model: SPLRewardModel,
        actor: nn.Module,
        critic1: nn.Module,
        critic2: nn.Module,
        actor_optim,
        critic1_optim,
        critic2_optim,
        tau: float = 0.005,
        gamma: float = 0.99,
        alpha: Union[float, Tuple] = 0.2
    ) -> None:
        super().__init__(
            dynamics,
            actor,
            critic1,
            critic2,
            actor_optim,
            critic1_optim,
            critic2_optim,
            tau=tau,
            gamma=gamma,
            alpha=alpha
        )
        self.reward_model = reward_model

    def rollout(
        self,
        init_obss: np.ndarray,
        rollout_length: int
    ) -> Tuple[Dict[str, np.ndarray], Dict]:

        num_transitions = 0
        rewards_arr = np.array([])
        rollout_transitions = defaultdict(list)

        observations = init_obss
        for _ in range(rollout_length):
            actions = self.select_action(observations)
            next_observations, _dyn_reward, terminals, info = self.dynamics.step(observations, actions)
            # Replace dynamics reward with pessimistic R_SPL
            rewards = self.reward_model.predict(observations, actions)  # (N, 1)

            rollout_transitions["obss"].append(observations)
            rollout_transitions["next_obss"].append(next_observations)
            rollout_transitions["actions"].append(actions)
            rollout_transitions["rewards"].append(rewards)
            rollout_transitions["terminals"].append(terminals)

            num_transitions += len(observations)
            rewards_arr = np.append(rewards_arr, rewards.flatten())

            nonterm_mask = (~terminals).flatten()
            if nonterm_mask.sum() == 0:
                break

            observations = next_observations[nonterm_mask]

        for k, v in rollout_transitions.items():
            rollout_transitions[k] = np.concatenate(v, axis=0)

        return rollout_transitions, \
            {"num_transitions": num_transitions, "reward_mean": rewards_arr.mean()}
