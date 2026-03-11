'''Hierarchical optimizer: micro + meso + macro at different intervals.'''

from __future__ import annotations

import random
from typing import Any

from pydantic import Field

from ..feedback.base import Trajectory
from .base import Optimizer, OptimizerConfig
from .population import Population
from .cmaes import CMAESConfig, CMAESOptimizer
from .mcts import MCTSConfig, MCTSOptimizer
from .prompt_breeding import PromptBreedingConfig, PromptBreedingOptimizer
from .evolutionary import GoalDrivenTopologyMutator


class HierarchicalConfig(OptimizerConfig):
    '''Configuration for the hierarchical optimizer.'''

    micro_config: CMAESConfig = Field(default_factory=CMAESConfig)
    meso_config: PromptBreedingConfig = Field(default_factory=PromptBreedingConfig)
    macro_config: MCTSConfig = Field(default_factory=MCTSConfig)
    meso_interval: int = Field(default=2, ge=1)
    macro_interval: int = Field(default=5, ge=1)
    goal_topo_model: str = Field(default='gpt-4o-mini')


class HierarchicalOptimizer(Optimizer):
    '''Recommended default: micro every gen, meso every meso_interval, macro every macro_interval.

    Micro (CMA-ES): continuous parameter tuning.
    Meso (PromptBreeding): prompt-level crossover.
    Macro (MCTS): topology search.
    '''

    def __init__(self, config: HierarchicalConfig, population: Population) -> None:
        super().__init__(config, population)
        self.config: HierarchicalConfig = config
        self.micro = CMAESOptimizer(config.micro_config, population)
        self.meso = PromptBreedingOptimizer(config.meso_config, population)
        self.macro = MCTSOptimizer(config.macro_config, population)
        self.goal_topo_mutator = GoalDrivenTopologyMutator()

    async def step(
        self,
        population: Population,
        fitness_scores: dict[str, float],
        trajectories: list[Trajectory],
    ) -> Population:
        gen = population.generation

        # Micro: every generation
        population = await self.micro.step(population, fitness_scores, trajectories)

        # Meso: every meso_interval generations
        if gen % self.config.meso_interval == 0:
            population = await self.meso.step(population, fitness_scores, trajectories)

        # Macro: every macro_interval generations
        if gen % self.config.macro_interval == 0:
            population = await self.macro.step(population, fitness_scores, trajectories)

            # Goal-driven structural mutation on one non-elite workflow
            elite_ids = {wf.workflow_id for wf in self._get_elite(population)}
            non_elite = [wf for wf in population.workflows if wf.workflow_id not in elite_ids]
            if non_elite:
                target = random.choice(non_elite)
                failures = [
                    f'Q:{t.question}|Pred:{t.prediction}|Truth:{t.ground_truth}'
                    for t in trajectories if t.env_reward < 0.5
                ][:3]
                mutated = await self.goal_topo_mutator.mutate(
                    target, self.config.goal_topo_model, failures, target.fitness
                )
                mutated.fitness = 0.0
                mutated.fitness_history = []
                new_wfs = [
                    mutated if wf.workflow_id == target.workflow_id else wf
                    for wf in population.workflows
                ]
                population.replace_workflows(new_wfs)

        return population

    def get_statistics(self) -> dict[str, Any]:
        return {
            'micro': self.micro.get_statistics(),
            'meso': self.meso.get_statistics(),
            'macro': self.macro.get_statistics(),
        }
