'''Bandit-style RL optimizer for SEMA.'''

from __future__ import annotations

import math
import random
from typing import Any

from pydantic import Field

from ..workflow.schema import Workflow
from ..feedback.base import Trajectory
from .base import Optimizer, OptimizerConfig
from .population import Population


class RLOptimizerConfig(OptimizerConfig):
    '''Configuration for RL (bandit-style) optimizer.'''

    learning_rate: float = Field(default=0.01, ge=0.0, le=1.0)
    discount_factor: float = Field(default=0.99, ge=0.0, le=1.0)
    policy_type: str = Field(default='bandit')  # "bandit" | "prompt_gradient"
    epsilon: float = Field(default=0.1, ge=0.0, le=1.0)
    ucb_c: float = Field(default=1.41, ge=0.0)
    reinforce_model: str = Field(default='gpt-4o-mini')


class RLOptimizer(Optimizer):
    '''Bandit-style RL optimizer: reward signals weight workflow selection.'''

    def __init__(self, config: RLOptimizerConfig, population: Population) -> None:
        super().__init__(config, population)
        self.config: RLOptimizerConfig = config
        self._arm_values: dict[str, float] = {}
        self._arm_visits: dict[str, int] = {}
        self._total_visits: int = 0
        self._stats: dict[str, Any] = {
            'policy_updates': 0,
            'explorations': 0,
            'exploitations': 0,
        }

    def _update_arms(self, fitness_scores: dict[str, float]) -> None:
        lr = self.config.learning_rate
        for wf_id, score in fitness_scores.items():
            if wf_id not in self._arm_values:
                self._arm_values[wf_id] = score
                self._arm_visits[wf_id] = 1
            else:
                # EMA update
                self._arm_values[wf_id] = (1 - lr) * self._arm_values[wf_id] + lr * score
                self._arm_visits[wf_id] += 1
            self._total_visits += 1

    def _ucb1_value(self, wf_id: str) -> float:
        visits = self._arm_visits.get(wf_id, 1)
        value = self._arm_values.get(wf_id, 0.0)
        if self._total_visits == 0:
            return value
        return value + self.config.ucb_c * math.sqrt(
            math.log(max(1, self._total_visits)) / visits
        )

    def _select_arm_bandit(self, workflows: list[Workflow]) -> Workflow:
        if random.random() < self.config.epsilon:
            self._stats['explorations'] += 1
            return random.choice(workflows)
        self._stats['exploitations'] += 1
        return max(workflows, key=lambda wf: self._ucb1_value(wf.workflow_id))

    def _perturb_prompts(self, workflow: Workflow) -> Workflow:
        '''Small textual perturbation: append/modify a random instruction hint.'''
        child = self._deep_copy_workflow(workflow)
        hints = [
            ' Be concise.',
            ' Think step by step.',
            ' Focus on accuracy.',
            ' Verify your answer.',
        ]
        for node in child.nodes:
            if random.random() < 0.3:
                node.action.instruction_prompt += random.choice(hints)
        return child

    async def _prompt_gradient_update(
        self, workflow: Workflow, advantage: float, positive_examples: list[str]
    ) -> Workflow:
        if advantage > 0:
            return self._perturb_prompts(workflow)

        if not positive_examples:
            return self._deep_copy_workflow(workflow)

        example_text = '\n'.join(positive_examples[:2])
        prompt = (
            f'These workflow prompts performed poorly. Here are better-performing examples:\n'
            f'{example_text}\n\n'
            f'Revise this workflow JSON to be more like the successful examples:\n'
            f'{workflow.to_json()}\n\n'
            'Return only the revised workflow JSON.'
        )
        import json as json_mod
        response = await self._call_llm(self.config.reinforce_model, prompt)
        if not response:
            return self._deep_copy_workflow(workflow)

        text = response.strip()
        if '```' in text:
            start = text.find('{', text.find('```'))
            end = text.rfind('}') + 1
            text = text[start:end] if start != -1 and end > start else text

        try:
            data = json_mod.loads(text)
            revised = workflow.__class__.model_validate(data)
            revised.parent_ids = [workflow.workflow_id]
            return revised
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

        self._update_arms(fitness_scores)
        self._stats['policy_updates'] += 1

        elite = self._get_elite(population)
        elite_ids = {wf.workflow_id for wf in elite}
        all_workflows = population.workflows

        new_workflows: list[Workflow] = [self._deep_copy_workflow(wf) for wf in elite]
        n_new = self.config.population_size - len(elite)

        if self.config.policy_type == 'prompt_gradient':
            avg_fitness = (
                sum(fitness_scores.values()) / len(fitness_scores) if fitness_scores else 0.0
            )
            positive_examples = [
                wf.to_json()
                for wf in sorted(all_workflows, key=lambda w: w.fitness, reverse=True)[:2]
            ]
            non_elite = [wf for wf in all_workflows if wf.workflow_id not in elite_ids]

            import asyncio
            async def update_one(wf: Workflow) -> Workflow:
                advantage = wf.fitness - avg_fitness
                return await self._prompt_gradient_update(wf, advantage, positive_examples)

            updated = await asyncio.gather(*[update_one(wf) for wf in non_elite[:n_new]])
            new_workflows.extend(updated)
        else:
            # Bandit policy
            for _ in range(n_new):
                selected = self._select_arm_bandit(all_workflows)
                child = self._perturb_prompts(selected)
                child.parent_ids = [selected.workflow_id]
                new_workflows.append(child)

        new_workflows = new_workflows[: self.config.population_size]
        population.replace_workflows(new_workflows)
        population.advance_generation()
        return population

    def get_statistics(self) -> dict[str, Any]:
        stats = dict(self._stats)
        stats['arm_values'] = dict(self._arm_values)
        return stats
