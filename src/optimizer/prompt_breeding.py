'''LLM-driven semantic prompt crossover (prompt breeding) optimizer.'''

from __future__ import annotations

import asyncio
import json
import random
from typing import Any

from pydantic import Field

from ..workflow.schema import Workflow
from ..feedback.base import Trajectory
from .base import Optimizer, OptimizerConfig
from .population import Population


class PromptBreedingConfig(OptimizerConfig):
    '''Configuration for prompt breeding optimizer.'''

    breeding_model: str = Field(default='gpt-4o-mini')
    num_failure_examples: int = Field(default=3, ge=1)


class PromptBreedingOptimizer(Optimizer):
    '''LLM-driven semantic prompt crossover between high-fitness workflows.'''

    def __init__(self, config: PromptBreedingConfig, population: Population) -> None:
        super().__init__(config, population)
        self.config: PromptBreedingConfig = config
        self._stats: dict[str, int] = {
            'breed_calls': 0,
            'breed_successes': 0,
        }

    def _get_prompts(self, workflow: Workflow) -> dict[str, dict[str, str]]:
        return {
            node.node_id: {
                'system_prompt': node.role.system_prompt,
                'instruction_prompt': node.action.instruction_prompt,
            }
            for node in workflow.nodes
        }

    def _get_failures(self, workflow_id: str, trajectories: list[Trajectory]) -> list[str]:
        failures = [
            f'Q: {t.question[:150]} | Pred: {t.prediction[:80]} | Truth: {t.ground_truth[:80]}'
            for t in trajectories
            if t.workflow_id == workflow_id and t.env_reward < 0.5
        ]
        return failures[: self.config.num_failure_examples]

    async def _breed(
        self,
        p1_prompts: dict[str, dict[str, str]],
        p2_prompts: dict[str, dict[str, str]],
        p1_failures: list[str],
        p2_failures: list[str],
        goal: str = '',
    ) -> dict[str, dict[str, str]]:
        '''Combine prompts from two parents, fixing their failures.'''
        self._stats['breed_calls'] += 1

        # Flatten prompts for the LLM
        p1_text = json.dumps(p1_prompts, indent=2)
        p2_text = json.dumps(p2_prompts, indent=2)
        failures_text = '\n'.join(
            [f'Parent A failure: {f}' for f in p1_failures]
            + [f'Parent B failure: {f}' for f in p2_failures]
        ) or 'No failures recorded.'
        goal_line = f'Experiment goal: {goal}\n\n' if goal else ''

        prompt = (
            f'You are a prompt optimizer. {goal_line}'
            'Combine the strengths of two prompt sets and fix their shared failures.\n\n'
            f'Parent A prompts:\n{p1_text}\n\n'
            f'Parent B prompts:\n{p2_text}\n\n'
            f'Failures to fix:\n{failures_text}\n\n'
            'Return a JSON object with the same node_id keys as Parent A, '
            'each having "system_prompt" and "instruction_prompt" fields. '
            'Return ONLY the JSON object.'
        )

        response = await self._call_llm(self.config.breeding_model, prompt)
        if not response:
            return p1_prompts

        text = response.strip()
        if '```' in text:
            start = text.find('{', text.find('```'))
            end = text.rfind('}') + 1
            text = text[start:end] if start != -1 and end > start else text

        try:
            result = json.loads(text)
            # Validate structure
            validated: dict[str, dict[str, str]] = {}
            for node_id, prompts in result.items():
                if isinstance(prompts, dict):
                    validated[node_id] = {
                        'system_prompt': str(prompts.get('system_prompt', '')),
                        'instruction_prompt': str(prompts.get('instruction_prompt', '')),
                    }
            self._stats['breed_successes'] += 1
            return validated if validated else p1_prompts
        except Exception:
            return p1_prompts

    def _apply_prompts(self, workflow: Workflow, prompts: dict[str, dict[str, str]]) -> Workflow:
        child = self._deep_copy_workflow(workflow)
        for node in child.nodes:
            if node.node_id in prompts:
                node_prompts = prompts[node.node_id]
                if 'system_prompt' in node_prompts:
                    node.role.system_prompt = node_prompts['system_prompt']
                if 'instruction_prompt' in node_prompts:
                    node.action.instruction_prompt = node_prompts['instruction_prompt']
        return child

    def _tournament_select_from_top(self, candidates: list[Workflow], k: int = 2) -> Workflow:
        tournament = random.sample(candidates, min(k, len(candidates)))
        return max(tournament, key=lambda wf: wf.fitness)

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

        all_workflows = sorted(population.workflows, key=lambda wf: wf.fitness, reverse=True)
        top_half = all_workflows[: max(2, len(all_workflows) // 2)]
        non_elite = [wf for wf in population.workflows if wf.workflow_id not in elite_ids]

        new_workflows: list[Workflow] = [self._deep_copy_workflow(wf) for wf in elite]

        async def breed_one(target: Workflow) -> Workflow:
            p1 = self._tournament_select_from_top(top_half)
            p2 = self._tournament_select_from_top(top_half)

            p1_prompts = self._get_prompts(p1)
            p2_prompts = self._get_prompts(p2)
            p1_failures = self._get_failures(p1.workflow_id, trajectories)
            p2_failures = self._get_failures(p2.workflow_id, trajectories)

            child_prompts = await self._breed(
                p1_prompts, p2_prompts, p1_failures, p2_failures, goal=p1.goal
            )

            # Apply to target structure (use p1 as structural template)
            child = self._apply_prompts(p1, child_prompts)
            child.parent_ids = [p1.workflow_id, p2.workflow_id]
            return child

        bred = await asyncio.gather(*[breed_one(wf) for wf in non_elite])
        new_workflows.extend(bred)
        new_workflows = new_workflows[: self.config.population_size]

        population.replace_workflows(new_workflows)
        population.advance_generation()
        return population

    def get_statistics(self) -> dict[str, Any]:
        return dict(self._stats)
