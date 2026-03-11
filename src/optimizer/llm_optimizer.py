'''OPRO-style LLM meta-optimizer for SEMA.'''

from __future__ import annotations

import asyncio
import json
from typing import Any

from pydantic import Field

from ..workflow.schema import Workflow
from ..feedback.base import Trajectory
from .base import Optimizer, OptimizerConfig
from .population import Population


class LLMOptimizerConfig(OptimizerConfig):
    '''Configuration for LLM-based (OPRO-style) optimizer.'''

    meta_llm_model: str = Field(default='gpt-4o-mini')
    num_candidates: int = Field(default=5, ge=1)
    context_window: int = Field(default=20, ge=1)


class LLMOptimizer(Optimizer):
    '''OPRO-style optimizer: meta-LLM generates improved workflows from history.'''

    def __init__(self, config: LLMOptimizerConfig, population: Population) -> None:
        super().__init__(config, population)
        self.config: LLMOptimizerConfig = config
        self._stats: dict[str, int] = {
            'calls_made': 0,
            'valid_proposals': 0,
            'accepted': 0,
        }

    def _build_meta_prompt(self, population: Population, trajectories: list[Trajectory]) -> str:
        # Sort trajectories by env_reward descending
        sorted_traj = sorted(trajectories, key=lambda t: t.env_reward, reverse=True)
        recent = sorted_traj[: self.config.context_window]

        # Build history of (workflow, fitness) from population
        wf_by_id = {wf.workflow_id: wf for wf in population.workflows}
        history_parts: list[str] = []
        for t in recent:
            wf = wf_by_id.get(t.workflow_id)
            if wf:
                history_parts.append(
                    f'Fitness: {t.env_reward:.3f}\n'
                    f'Workflow JSON:\n{wf.to_json()}'
                )

        history_text = '\n\n---\n\n'.join(history_parts) if history_parts else 'No history yet.'

        goal = next((wf.goal for wf in population.workflows if wf.goal), '')
        goal_line = f'Experiment goal: {goal}\n\n' if goal else ''

        # Check if population has blank workflows
        has_nodes = any(wf.nodes for wf in population.workflows)
        if not has_nodes:
            instruction = (
                'Bootstrap a new workflow JSON with at least 2 nodes. '
                'Each node must have role (name, system_prompt) and action (name, instruction_prompt). '
                'Tailor role names and prompts to the experiment goal. '
                'Return ONLY valid JSON that matches the Workflow schema.'
            )
        else:
            instruction = (
                'Propose an improved workflow JSON with better prompts and/or topology. '
                'You may add, remove, or rename nodes and change role names to better serve the goal. '
                'Return ONLY valid JSON that matches the Workflow schema.'
            )

        return (
            f'You are a meta-optimizer improving multi-agent workflows.\n\n'
            f'{goal_line}'
            f'Past workflow performances (best first):\n\n{history_text}\n\n'
            f'{instruction}'
        )

    async def _generate_candidate(self, prompt: str, generation: int, parent_ids: list[str]) -> Workflow | None:
        self._stats['calls_made'] += 1
        response = await self._call_llm(self.config.meta_llm_model, prompt)
        if not response:
            return None

        # Extract JSON from response (handle markdown code blocks)
        text = response.strip()
        if '```' in text:
            start = text.find('{', text.find('```'))
            end = text.rfind('}') + 1
            text = text[start:end] if start != -1 and end > start else text

        try:
            data = json.loads(text)
            wf = Workflow.model_validate(data)
            wf.generation = generation
            wf.parent_ids = parent_ids
            self._stats['valid_proposals'] += 1
            return wf
        except Exception:
            return None

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
        parent_ids = [wf.workflow_id for wf in elite[:2]]

        prompt = self._build_meta_prompt(population, trajectories)
        next_gen = population.generation + 1

        # Generate candidates concurrently
        candidates_raw = await asyncio.gather(
            *[self._generate_candidate(prompt, next_gen, parent_ids)
              for _ in range(self.config.num_candidates)]
        )
        candidates = [c for c in candidates_raw if c is not None]

        # Replace worst non-elite slots with valid candidates
        all_workflows = population.workflows
        non_elite = sorted(
            [wf for wf in all_workflows if wf.workflow_id not in elite_ids],
            key=lambda wf: wf.fitness,
        )

        new_workflows = list(elite)
        slots_to_fill = self.config.population_size - len(elite)
        replacements = candidates[:slots_to_fill]
        remaining_non_elite = non_elite[len(replacements):]

        for wf in replacements:
            new_workflows.append(wf)
            self._stats['accepted'] += 1

        new_workflows.extend([self._deep_copy_workflow(wf) for wf in remaining_non_elite])
        new_workflows = new_workflows[: self.config.population_size]

        population.replace_workflows(new_workflows)
        population.advance_generation()
        return population

    def get_statistics(self) -> dict[str, Any]:
        return dict(self._stats)
