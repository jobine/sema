'''Textual gradient descent optimizer for SEMA.'''

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import Field

from ..workflow.schema import Workflow
from ..feedback.base import Trajectory
from .base import Optimizer, OptimizerConfig
from .population import Population


class TextGradConfig(OptimizerConfig):
    '''Configuration for textual gradient descent optimizer.'''

    gradient_model: str = Field(default='gpt-4o-mini')
    step_size: str = Field(default='medium')  # "small" | "medium" | "large"


class TextGradOptimizer(Optimizer):
    '''Textual gradient descent on workflow prompts.'''

    def __init__(self, config: TextGradConfig, population: Population) -> None:
        super().__init__(config, population)
        self.config: TextGradConfig = config
        self._stats: dict[str, int] = {
            'gradients_computed': 0,
            'prompts_updated': 0,
        }

    def _get_failed_trajectories(self, workflow_id: str, trajectories: list[Trajectory]) -> list[Trajectory]:
        return [
            t for t in trajectories
            if t.workflow_id == workflow_id and t.ground_truth and t.prediction != t.ground_truth
        ]

    async def _compute_gradient(self, trajectory: Trajectory) -> str:
        '''Ask the LLM what changes to make to fix this failure.'''
        prompt = (
            f'Prediction: {trajectory.prediction}\n'
            f'Ground truth: {trajectory.ground_truth}\n\n'
            f'Identify what the system_prompt and instruction_prompt should change '
            f'to fix this error. Be {self.config.step_size} in the scope of changes suggested. '
            'Describe the direction of change concisely (1-3 sentences).'
        )
        self._stats['gradients_computed'] += 1
        return await self._call_llm(self.config.gradient_model, prompt)

    async def _apply_gradient(self, workflow: Workflow, gradient: str) -> Workflow:
        '''Apply gradient: ask LLM to rewrite prompts guided by gradient text.'''
        child = self._deep_copy_workflow(workflow)
        if not child.nodes or not gradient:
            return child

        import asyncio as _asyncio
        import json as json_mod

        async def update_node(node_idx: int) -> None:
            node = child.nodes[node_idx]
            prompt = (
                f'Gradient direction: {gradient}\n\n'
                f'Current system_prompt:\n{node.role.system_prompt}\n\n'
                f'Current instruction_prompt:\n{node.action.instruction_prompt}\n\n'
                'Rewrite both prompts following the gradient direction. '
                'Return JSON with keys "system_prompt" and "instruction_prompt".'
            )
            response = await self._call_llm(self.config.gradient_model, prompt)
            if response:
                text = response.strip()
                if '```' in text:
                    start = text.find('{', text.find('```'))
                    end = text.rfind('}') + 1
                    text = text[start:end] if start != -1 and end > start else text
                try:
                    data = json_mod.loads(text)
                    if 'system_prompt' in data:
                        node.role.system_prompt = str(data['system_prompt'])
                    if 'instruction_prompt' in data:
                        node.action.instruction_prompt = str(data['instruction_prompt'])
                    self._stats['prompts_updated'] += 1
                except Exception:
                    pass

        await _asyncio.gather(*[update_node(i) for i in range(len(child.nodes))])
        child.parent_ids = [workflow.workflow_id]
        return child

    async def _improve_workflow(self, workflow: Workflow, trajectories: list[Trajectory]) -> Workflow:
        failures = self._get_failed_trajectories(workflow.workflow_id, trajectories)
        if not failures:
            return self._deep_copy_workflow(workflow)

        # Aggregate gradients from failures
        gradients = await asyncio.gather(*[self._compute_gradient(t) for t in failures[:3]])
        combined_gradient = ' '.join(g for g in gradients if g)

        if not combined_gradient:
            return self._deep_copy_workflow(workflow)

        return await self._apply_gradient(workflow, combined_gradient)

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

        improved = await asyncio.gather(
            *[self._improve_workflow(wf, trajectories) for wf in non_elite]
        )

        new_workflows = [self._deep_copy_workflow(wf) for wf in elite] + list(improved)
        new_workflows = new_workflows[: self.config.population_size]

        population.replace_workflows(new_workflows)
        population.advance_generation()
        return population

    def get_statistics(self) -> dict[str, Any]:
        return dict(self._stats)
