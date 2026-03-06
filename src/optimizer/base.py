'''Abstract base classes for SEMA optimizers.'''

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..workflow.schema import Workflow
from ..feedback.base import Trajectory
from .population import Population


class OptimizerConfig(BaseModel):
    '''Base configuration for all optimizers.'''

    model_config = ConfigDict(extra='allow')

    population_size: int = Field(default=10, ge=1, description='Number of workflows in population')
    max_generations: int = Field(default=50, ge=1, description='Maximum number of generations')
    elitism_rate: float = Field(default=0.2, ge=0.0, le=1.0, description='Fraction of elite workflows to preserve')


class Optimizer(ABC):
    '''Abstract base class for all workflow optimizers.'''

    def __init__(self, config: OptimizerConfig, population: Population) -> None:
        self.config = config
        self.population = population

    @abstractmethod
    async def step(
        self,
        population: Population,
        fitness_scores: dict[str, float],
        trajectories: list[Trajectory],
    ) -> Population:
        '''Execute one optimization step and return updated population.'''
        ...

    @abstractmethod
    def get_statistics(self) -> dict[str, Any]:
        '''Return optimizer-specific statistics.'''
        ...

    def _get_elite(self, population: Population) -> list[Workflow]:
        '''Return elite workflows based on config elitism_rate.'''
        n = math.ceil(self.config.elitism_rate * self.config.population_size)
        return population.get_elite(n)

    def _deep_copy_workflow(self, wf: Workflow) -> Workflow:
        '''Deep copy a workflow via serialization roundtrip.'''
        return Workflow.from_json(wf.to_json())

    async def _call_llm(self, model: str, prompt: str) -> str:
        '''Call an LLM with error handling. Returns empty string on failure.'''
        try:
            from ..models.models import AsyncLLM
            llm = AsyncLLM(model)
            result = await llm(prompt)
            return result if isinstance(result, str) else str(result)
        except Exception:
            return ''
