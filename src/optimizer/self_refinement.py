'''Self-refinement optimizer: each workflow critiques its failures and self-improves.'''

from __future__ import annotations

import asyncio
import json
from typing import Any

from pydantic import Field

from ..workflow.schema import Workflow
from ..feedback.base import Trajectory
from .base import Optimizer, OptimizerConfig
from .population import Population


class SelfRefinementConfig(OptimizerConfig):
    '''Configuration for self-refinement optimizer.'''

    max_reflection_rounds: int = Field(default=3, ge=1)
    critique_model: str = Field(default='gpt-4o-mini')


class SelfRefinementOptimizer(Optimizer):
    '''Each workflow self-critiques its failed trajectories and improves.'''

    def __init__(self, config: SelfRefinementConfig, population: Population) -> None:
        super().__init__(config, population)
        self.config: SelfRefinementConfig = config
        self._stats: dict[str, int] = {
            'refinements_attempted': 0,
            'refinements_succeeded': 0,
        }

    def _get_failures(self, workflow_id: str, trajectories: list[Trajectory]) -> list[Trajectory]:
        return [t for t in trajectories if t.workflow_id == workflow_id and t.env_reward < 0.5]

    def _build_critique_prompt(self, workflow: Workflow, failures: list[Trajectory]) -> str:
        failure_text = '\n'.join(
            f'- Q: {t.question[:200]} | Pred: {t.prediction[:100]} | Truth: {t.ground_truth[:100]}'
            for t in failures[: self.config.max_reflection_rounds]
        ) or 'No specific failures recorded.'

        return (
            'You are a self-critiquing optimizer. Analyze this workflow and its failures, '
            'then produce an improved version.\n\n'
            f'Current workflow JSON:\n{workflow.to_json()}\n\n'
            f'Failed cases:\n{failure_text}\n\n'
            'Provide a self-critique and return an improved workflow JSON. '
            'Return ONLY the improved workflow JSON (no extra text).'
        )

    async def _refine_workflow(self, workflow: Workflow, trajectories: list[Trajectory]) -> Workflow:
        failures = self._get_failures(workflow.workflow_id, trajectories)
        if not failures:
            return self._deep_copy_workflow(workflow)

        self._stats['refinements_attempted'] += 1
        prompt = self._build_critique_prompt(workflow, failures)
        response = await self._call_llm(self.config.critique_model, prompt)

        if not response:
            return self._deep_copy_workflow(workflow)

        text = response.strip()
        if '```' in text:
            start = text.find('{', text.find('```'))
            end = text.rfind('}') + 1
            text = text[start:end] if start != -1 and end > start else text

        try:
            data = json.loads(text)
            refined = Workflow.model_validate(data)
            refined.generation = workflow.generation + 1
            refined.parent_ids = [workflow.workflow_id]
            self._stats['refinements_succeeded'] += 1
            return refined
        except Exception:
            return self._deep_copy_workflow(workflow)

    async def step(
        self,
        population: Population,
        fitness_scores: dict[str, float],
        trajectories: list[Trajectory],
    ) -> Population:
        for wf_id, score in fitness_scores.items():
            population.update_fitness(wf_id, score)

        elite = self._get_elite(population)
        elite_ids = {wf.workflow_id for wf in elite}
        non_elite = [wf for wf in population.workflows if wf.workflow_id not in elite_ids]

        refined = await asyncio.gather(*[self._refine_workflow(wf, trajectories) for wf in non_elite])

        new_workflows = [self._deep_copy_workflow(wf) for wf in elite] + list(refined)
        new_workflows = new_workflows[: self.config.population_size]

        population.replace_workflows(new_workflows)
        population.advance_generation()
        return population

    def get_statistics(self) -> dict[str, Any]:
        return dict(self._stats)
