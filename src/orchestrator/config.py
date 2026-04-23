'''SEMAConfig: experiment configuration for the SEMA orchestrator.'''

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ..config.paths import SEMAPaths
from ..workflow.environment import Environment
from ..optimizer.base import OptimizerConfig
from ..feedback.meta_reward import MetaRewardConfig


class SEMAConfig(BaseModel):
    model_config = ConfigDict(extra='allow')

    experiment_name: str = 'sema_experiment'
    storage_root: str = Field(default_factory=lambda: str(SEMAPaths.load().experiments))
    goal: str = ''
    environment: Environment
    optimizer_type: str = 'hierarchical'
    optimizer_config: OptimizerConfig = OptimizerConfig()
    seed_template: str = 'auto'            # auto|blank|single_agent|chain|debate|hierarchical
    bootstrap_model: str = 'gpt-4o-mini'  # model used to design the seed workflow when seed_template='auto'
    executor_model: str = 'gpt-4o-mini'   # default model for workflow node agents during execution
    optimizer_model: str = 'gpt-4o-mini'  # default model for all optimizer LLM calls
    population_size: int = 10
    eval_samples_per_generation: int = 50
    eval_dataset: str = 'validate'        # train|validate|test
    meta_reward: MetaRewardConfig = MetaRewardConfig()
    max_generations: int = 50
    early_stop_generations: int = 10
    fitness_target: float | None = None
