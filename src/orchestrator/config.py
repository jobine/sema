'''SEMAConfig: experiment configuration for the SEMA orchestrator.'''

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from ..workflow.environment import Environment
from ..optimizer.base import OptimizerConfig
from ..feedback.meta_reward import MetaRewardConfig


class SEMAConfig(BaseModel):
    model_config = ConfigDict(extra='allow')

    experiment_name: str = 'sema_experiment'
    storage_root: str = '~/.sema/experiments'
    environment: Environment
    optimizer_type: str = 'hierarchical'
    optimizer_config: OptimizerConfig = OptimizerConfig()
    seed_template: str = 'single_agent'   # blank|single_agent|chain|debate|hierarchical
    population_size: int = 10
    eval_samples_per_generation: int = 50
    eval_dataset: str = 'validate'        # train|validate|test
    meta_reward: MetaRewardConfig = MetaRewardConfig()
    max_generations: int = 50
    early_stop_generations: int = 10
    fitness_target: float | None = None
