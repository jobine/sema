'''ExperimentTracker: checkpointing and history logging for SEMA experiments.

Storage layout:
    {storage_root}/{experiment_name}/
    ├── config.json
    ├── environment.json
    ├── history.jsonl
    ├── checkpoints/
    │   ├── gen_000.jsonl
    │   └── gen_001.jsonl
    ├── best_workflow.json
    └── report.md
'''

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..workflow.schema import Workflow
from ..optimizer.population import Population
from ..workflow.environment import Environment


class ExperimentTracker:
    '''Tracks experiment history and manages checkpoints.'''

    def __init__(
        self,
        experiment_name: str,
        storage_root: str = '~/.sema/experiments',
    ) -> None:
        self._experiment_name = experiment_name
        self._root = Path(storage_root).expanduser() / experiment_name
        self._checkpoints_dir = self._root / 'checkpoints'
        self._history_path = self._root / 'history.jsonl'
        self._best_workflow_path = self._root / 'best_workflow.json'
        self._config_path = self._root / 'config.json'
        self._env_path = self._root / 'environment.json'
        self._report_path = self._root / 'report.md'

        self._root.mkdir(parents=True, exist_ok=True)
        self._checkpoints_dir.mkdir(parents=True, exist_ok=True)

        self._history: list[dict[str, Any]] = []
        self._best_fitness: float = 0.0

        if self._history_path.exists():
            with open(self._history_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self._history.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            if self._history:
                self._best_fitness = max(
                    h.get('best_fitness', 0.0) for h in self._history
                )

    async def log_generation(
        self,
        generation: int,
        stats: dict[str, Any],
        fitness_scores: dict[str, float],
        best_workflow: Workflow,
    ) -> None:
        '''Append one JSONL line to history.jsonl; update best_workflow.json if improved.'''
        entry: dict[str, Any] = {
            'generation': generation,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            **stats,
            'fitness_scores': fitness_scores,
            'best_workflow_id': best_workflow.workflow_id,
        }
        self._history.append(entry)
        with open(self._history_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + '\n')

        gen_best = stats.get('best_fitness', 0.0)
        if gen_best > self._best_fitness:
            self._best_fitness = gen_best
            with open(self._best_workflow_path, 'w', encoding='utf-8') as f:
                f.write(best_workflow.to_json())

    async def save_checkpoint(
        self,
        population: Population,
        environment: Environment,
        generation: int,
    ) -> None:
        '''Write population to checkpoints/gen_{generation:03d}.jsonl and environment.json.'''
        checkpoint_path = self._checkpoints_dir / f'gen_{generation:03d}.jsonl'
        await population.save(checkpoint_path)
        with open(self._env_path, 'w', encoding='utf-8') as f:
            f.write(environment.model_dump_json())

    async def load_checkpoint(
        self, generation: int | None = None
    ) -> tuple[Population, Environment, int]:
        '''Load checkpoint from disk.

        Args:
            generation: Specific generation to load; None loads the latest.

        Returns:
            (population, environment, generation_number)
        '''
        checkpoints = sorted(self._checkpoints_dir.glob('gen_*.jsonl'))
        if not checkpoints:
            raise FileNotFoundError(
                f'No checkpoints found in {self._checkpoints_dir}'
            )

        if generation is not None:
            checkpoint_path = self._checkpoints_dir / f'gen_{generation:03d}.jsonl'
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f'Checkpoint for generation {generation} not found: {checkpoint_path}'
                )
            gen_number = generation
        else:
            checkpoint_path = checkpoints[-1]
            gen_number = int(checkpoint_path.stem.split('_')[1])

        population = await Population.load(checkpoint_path)

        env = Environment(name='unknown')
        if self._env_path.exists():
            with open(self._env_path, 'r', encoding='utf-8') as f:
                env = Environment(**json.loads(f.read()))

        return population, env, gen_number

    async def save_config(self, config: Any) -> None:
        '''Write config.json (any Pydantic model).'''
        with open(self._config_path, 'w', encoding='utf-8') as f:
            f.write(config.model_dump_json(indent=2))

    def get_history(self) -> list[dict[str, Any]]:
        '''Return list of per-generation stat dicts.'''
        return list(self._history)

    def summary_report(self) -> str:
        '''Generate a Markdown report, write report.md, and return the content.'''
        generations_run = len(self._history)
        best_wf_id = ''
        for entry in self._history:
            if abs(entry.get('best_fitness', 0.0) - self._best_fitness) < 1e-9:
                best_wf_id = entry.get('best_workflow_id', '')
                break

        rows = [
            f'| {h.get("generation", "?")} '
            f'| {h.get("best_fitness", 0.0):.4f} '
            f'| {h.get("avg_fitness", 0.0):.4f} |'
            for h in self._history
        ]
        table = '\n'.join(rows)

        report = (
            f'# SEMA Experiment Report: {self._experiment_name}\n\n'
            f'## Summary\n'
            f'- **Generations run**: {generations_run}\n'
            f'- **Best fitness**: {self._best_fitness:.4f}\n'
            f'- **Best workflow ID**: {best_wf_id}\n\n'
            f'## Improvement Curve\n\n'
            f'| Generation | Best Fitness | Avg Fitness |\n'
            f'|------------|--------------|-------------|\n'
            f'{table}\n'
        )

        with open(self._report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        return report
