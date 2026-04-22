'''SEMAOrchestrator: main self-evolving loop wiring evaluation, logging, and optimization.'''

from __future__ import annotations

import asyncio
import random
from typing import Any

from .config import SEMAConfig
from .experiment import ExperimentTracker
from ..workflow.executor import WorkflowExecutor
from ..workflow.templates import WorkflowTemplate
from ..optimizer.population import Population
from ..optimizer.registry import OptimizerRegistry
from ..feedback.base import Trajectory, FeedbackCollector
from ..feedback.meta_reward import MetaRewardComputer
from ..models.model_usage import ModelUsage
from ..utils import get_logger

logger = get_logger(__name__)


class SEMAOrchestrator:
    '''Orchestrates the full SEMA self-evolution loop.

    Usage:
        config = SEMAConfig(environment=env)
        orch = SEMAOrchestrator(config)
        result = await orch.run()
    '''

    def __init__(self, config: SEMAConfig) -> None:
        self.config = config
        self.tracker = ExperimentTracker(config.experiment_name, config.storage_root)
        self._benchmark: Any = None

    # -----------------------------------------------------------------------
    # Public entry points
    # -----------------------------------------------------------------------

    async def run(self) -> dict[str, Any]:
        '''Full run from generation 0.'''
        self._benchmark = self.config.environment.get_benchmark()
        self._benchmark.load_data()
        population = await self._build_seed_population()
        self._propagate_optimizer_model()
        optimizer = OptimizerRegistry.create(
            self.config.optimizer_type, self.config.optimizer_config, population
        )
        await self.tracker.save_config(self.config)
        return await self._run_loop(population, optimizer, start_gen=0)

    async def resume(self, checkpoint_generation: int | None = None) -> dict[str, Any]:
        '''Load a checkpoint and continue the loop.'''
        population, saved_env, generation = await self.tracker.load_checkpoint(
            checkpoint_generation
        )
        self._benchmark = self.config.environment.get_benchmark()
        self._benchmark.load_data()
        if self.config.environment.has_changed(saved_env):
            for wf in population.workflows:
                wf.fitness = 0.0
                wf.fitness_history = []
        self._propagate_optimizer_model()
        optimizer = OptimizerRegistry.create(
            self.config.optimizer_type, self.config.optimizer_config, population
        )
        return await self._run_loop(population, optimizer, start_gen=generation + 1)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _propagate_optimizer_model(self) -> None:
        '''Set optimizer_model on the optimizer config from the top-level SEMAConfig.'''
        self.config.optimizer_config.optimizer_model = self.config.optimizer_model

    async def _run_loop(
        self, population: Population, optimizer: Any, start_gen: int
    ) -> dict[str, Any]:
        feedback_collector = FeedbackCollector()
        # Restore best fitness from tracker so early-stop is correct on resume
        best_fitness = self.tracker._best_fitness
        no_improve = 0
        generation = start_gen - 1
        # Track how many history entries existed before this session
        history_offset = len(self.tracker._history)

        dataset_items: list[dict] = (
            getattr(self._benchmark, f'{self.config.eval_dataset}_data', None) or []
        )

        for generation in range(start_gen, self.config.max_generations):
            # A: Evaluate
            fitness_scores, trajectories = await self._evaluate_generation(
                population, self._benchmark, dataset_items
            )
            for wf_id, score in fitness_scores.items():
                population.update_fitness(wf_id, score)
            for t in trajectories:
                feedback_collector.record_trajectory(t)

            # B: Log
            stats = population.summary()
            best = population.best_workflow
            await self.tracker.log_generation(generation, stats, fitness_scores, best)
            await self.tracker.save_checkpoint(
                population, self.config.environment, generation
            )

            # C: Early stop
            if stats['best_fitness'] > best_fitness:
                best_fitness = stats['best_fitness']
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= self.config.early_stop_generations:
                break
            if self.config.fitness_target is not None and best_fitness >= self.config.fitness_target:
                break

            # D: Optimize
            population = await optimizer.step(population, fitness_scores, trajectories)

        # Model usage summary
        model_usage = ModelUsage.get_instance()
        logger.info('\n' + model_usage.summary())

        report = self.tracker.summary_report(extra_sections=model_usage.report_section())
        return {
            'best_workflow': population.best_workflow,
            'best_fitness': best_fitness,
            'generations': generation + 1,
            'report': report,
            'history': self.tracker.get_history()[history_offset:],
        }

    async def _evaluate_generation(
        self,
        population: Population,
        benchmark: Any,
        dataset_items: list[dict],
    ) -> tuple[dict[str, float], list[Trajectory]]:
        '''Evaluate all workflows concurrently per workflow, sequentially across workflows.'''
        meta_reward_computer = MetaRewardComputer(self.config.meta_reward)
        fitness_scores: dict[str, float] = {}
        all_trajectories: list[Trajectory] = []

        for workflow in population.workflows:
            samples = self._sample_dataset(
                dataset_items, self.config.eval_samples_per_generation
            )

            async def eval_one(item: dict, wf=workflow) -> Trajectory:
                task = {
                    'question': item['question'],
                    'context': self._stringify_context(item.get('context', '')),
                    'answer_format': getattr(benchmark, 'answer_format', ''),
                }
                executor = WorkflowExecutor(default_model=self.config.executor_model)
                result = await executor.execute(wf, task)
                ground_truth = item.get('answer', item.get('label', ''))
                eval_result = await benchmark.evaluate(result.answer, str(ground_truth))
                t = Trajectory(
                    workflow_id=wf.workflow_id,
                    question=item['question'],
                    context=task['context'],
                    steps=result.execution_trace,
                    prediction=result.answer,
                    ground_truth=str(ground_truth),
                    env_reward=eval_result.get('f1', 0.0),
                    node_outputs=result.node_outputs,
                    total_llm_calls=result.total_llm_calls,
                )
                t.meta_reward = meta_reward_computer.compute(t)
                return t

            wf_trajectories: tuple[Trajectory, ...] = await asyncio.gather(
                *[eval_one(item) for item in samples]
            )
            fitness = (
                sum(t.meta_reward for t in wf_trajectories) / len(wf_trajectories)
                if wf_trajectories
                else 0.0
            )
            fitness_scores[workflow.workflow_id] = fitness
            all_trajectories.extend(wf_trajectories)

        return fitness_scores, all_trajectories

    def _sample_dataset(self, dataset_items: list[dict], n: int) -> list[dict]:
        '''Return up to n randomly sampled items; returns all shuffled if n >= len.'''
        if not dataset_items:
            return []
        shuffled = list(dataset_items)
        random.shuffle(shuffled)
        return shuffled[:n]

    def _stringify_context(self, context: Any) -> str:
        '''Convert HotpotQA-style list-of-[title, sentences] to plain text.'''
        if not context:
            return ''
        if isinstance(context, str):
            return context
        parts: list[str] = []
        for item in context:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                title = item[0]
                sentences = item[1]
                text = ' '.join(sentences) if isinstance(sentences, list) else str(sentences)
                parts.append(f'{title}: {text}')
            elif isinstance(item, str):
                parts.append(item)
        return '\n'.join(parts)

    async def _build_seed_population(self) -> Population:
        '''Build a seed population from the configured template.'''
        if self.config.seed_template == 'auto':
            seed_workflow = await WorkflowTemplate.from_goal(
                self.config.goal,
                self.config.bootstrap_model,
                self.config.environment.name,
            )
        else:
            template_map = {
                'blank': WorkflowTemplate.blank,
                'single_agent': WorkflowTemplate.single_agent,
                'chain': WorkflowTemplate.chain,
                'debate': WorkflowTemplate.debate,
                'hierarchical': WorkflowTemplate.hierarchical,
            }
            factory = template_map.get(self.config.seed_template, WorkflowTemplate.single_agent)
            seed_workflow = factory(goal=self.config.goal, environment=self.config.environment.name)

        population = Population(
            population_size=self.config.population_size,
            elitism_rate=self.config.optimizer_config.elitism_rate,
        )
        population.initialize([seed_workflow])
        return population
