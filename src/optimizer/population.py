'''Population manager for SEMA evolutionary optimization.'''

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

from ..workflow.schema import Workflow


class Population:
    '''Manages a population of Workflow objects for evolutionary optimization.'''

    def __init__(self, population_size: int = 10, elitism_rate: float = 0.2) -> None:
        self._size = population_size
        self._elitism_rate = elitism_rate
        self._workflows: list[Workflow] = []
        self._generation: int = 0

    def initialize(self, seed_workflows: list[Workflow] | None = None) -> None:
        '''Initialize population from seed workflows or create blank Workflows.

        Each copy gets a fresh workflow_id to avoid ID collisions.
        '''
        import uuid as _uuid
        self._workflows = []
        if seed_workflows:
            for i in range(self._size):
                source = seed_workflows[i % len(seed_workflows)]
                copy = Workflow.from_json(source.to_json())
                copy.workflow_id = str(_uuid.uuid4())
                copy.fitness = 0.0
                copy.fitness_history = []
                self._workflows.append(copy)
        else:
            for _ in range(self._size):
                self._workflows.append(Workflow())

    @property
    def workflows(self) -> list[Workflow]:
        return list(self._workflows)

    @property
    def generation(self) -> int:
        '''Max generation number across all workflows.'''
        if not self._workflows:
            return self._generation
        return max(self._generation, max(wf.generation for wf in self._workflows))

    @property
    def best_workflow(self) -> Workflow:
        '''Workflow with highest fitness.'''
        if not self._workflows:
            raise ValueError('Population is empty')
        return max(self._workflows, key=lambda wf: wf.fitness)

    def get_elite(self, n: int | None = None) -> list[Workflow]:
        '''Return top-n workflows sorted by fitness descending.

        n defaults to ceil(elitism_rate * population_size).
        '''
        if n is None:
            n = math.ceil(self._elitism_rate * self._size)
        sorted_wf = sorted(self._workflows, key=lambda wf: wf.fitness, reverse=True)
        return sorted_wf[:n]

    def update_fitness(self, workflow_id: str, fitness: float) -> None:
        '''Set fitness for a workflow and record in its history.'''
        for wf in self._workflows:
            if wf.workflow_id == workflow_id:
                wf.fitness = fitness
                wf.fitness_history.append(fitness)
                return

    def advance_generation(self) -> None:
        '''Increment generation counter and update all workflow generation numbers.'''
        self._generation += 1
        for wf in self._workflows:
            wf.generation = self._generation

    def replace_workflows(self, new_workflows: list[Workflow]) -> None:
        '''Replace the internal workflow list with a new one.'''
        self._workflows = list(new_workflows)

    async def save(self, path: Path) -> None:
        '''Write population to a JSONL file (one Workflow per line).'''
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            for wf in self._workflows:
                f.write(wf.to_json().replace('\n', ' ') + '\n')

    @classmethod
    async def load(cls, path: Path) -> Population:
        '''Load population from a JSONL file.'''
        path = Path(path)
        pop = cls()
        workflows: list[Workflow] = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    workflows.append(Workflow.from_json(line))
        pop._workflows = workflows
        pop._size = len(workflows)
        if workflows:
            pop._generation = max(wf.generation for wf in workflows)
        return pop

    def summary(self) -> dict[str, Any]:
        '''Return summary statistics of the population.'''
        if not self._workflows:
            return {
                'generation': self._generation,
                'size': 0,
                'best_fitness': 0.0,
                'avg_fitness': 0.0,
                'worst_fitness': 0.0,
                'diversity': 0.0,
            }
        fitnesses = [wf.fitness for wf in self._workflows]
        n = len(fitnesses)
        avg = sum(fitnesses) / n
        variance = sum((f - avg) ** 2 for f in fitnesses) / n
        diversity = variance ** 0.5
        return {
            'generation': self.generation,
            'size': n,
            'best_fitness': max(fitnesses),
            'avg_fitness': avg,
            'worst_fitness': min(fitnesses),
            'diversity': diversity,
        }
