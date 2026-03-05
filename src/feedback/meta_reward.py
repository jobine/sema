'''Meta-reward computation for SEMA feedback system.'''

from __future__ import annotations

from pydantic import BaseModel, Field, ConfigDict

from .base import Trajectory


class MetaRewardConfig(BaseModel):
    '''Configuration for meta-reward computation weights.'''

    model_config = ConfigDict(validate_assignment=True)

    accuracy_weight: float = Field(default=0.4, ge=0.0, le=1.0, description='Weight for accuracy (env_reward)')
    efficiency_weight: float = Field(default=0.3, ge=0.0, le=1.0, description='Weight for efficiency')
    tool_use_weight: float = Field(default=0.15, ge=0.0, le=1.0, description='Weight for tool usage quality')
    memory_use_weight: float = Field(default=0.15, ge=0.0, le=1.0, description='Weight for memory usage quality')


class MetaRewardComputer:
    '''Computes meta-rewards from execution trajectories.

    Meta-reward is a weighted combination of:
    - Accuracy: directly from env_reward
    - Efficiency: inverse of steps taken (fewer steps = better)
    - Tool quality: ratio of successful tool calls
    - Memory quality: ratio of memory retrievals that contributed to the answer
    '''

    def __init__(self, config: MetaRewardConfig | None = None) -> None:
        self.config = config or MetaRewardConfig()

    def compute(self, trajectory: Trajectory) -> float:
        '''Compute the meta-reward for a single trajectory.'''
        accuracy = trajectory.env_reward

        # Efficiency: inverse of steps, normalized to [0, 1]
        num_steps = max(len(trajectory.steps), 1)
        efficiency = 1.0 / num_steps

        # Tool quality: ratio of successful tool calls
        tool_calls = trajectory.tool_calls
        if tool_calls:
            successful = sum(1 for tc in tool_calls if tc.get('success', False))
            tool_quality = successful / len(tool_calls)
        else:
            tool_quality = 0.5  # Neutral if no tools used

        # Memory quality: ratio of retrievals with non-empty content
        retrievals = trajectory.memory_retrievals
        if retrievals:
            useful = sum(1 for r in retrievals if r.get('content', ''))
            memory_quality = useful / len(retrievals)
        else:
            memory_quality = 0.5  # Neutral if no memory used

        meta_reward = (
            self.config.accuracy_weight * accuracy
            + self.config.efficiency_weight * efficiency
            + self.config.tool_use_weight * tool_quality
            + self.config.memory_use_weight * memory_quality
        )

        return meta_reward

    def compute_batch(self, trajectories: list[Trajectory]) -> list[float]:
        '''Compute meta-rewards for a batch of trajectories.'''
        return [self.compute(t) for t in trajectories]
