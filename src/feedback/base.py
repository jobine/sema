'''Base classes for the SEMA feedback system.'''

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ConfigDict


class Trajectory(BaseModel):
    '''Records a complete execution trajectory for feedback analysis.'''

    model_config = ConfigDict(validate_assignment=True)

    workflow_id: str = Field(default='', description='Workflow that produced this trajectory')
    question: str = Field(default='', description='Input question')
    context: str = Field(default='', description='Input context')
    steps: list[dict[str, Any]] = Field(default_factory=list, description='Reasoning steps taken')
    memory_retrievals: list[dict[str, Any]] = Field(default_factory=list, description='Memory retrieval records')
    tool_calls: list[dict[str, Any]] = Field(default_factory=list, description='Tool call records')
    prediction: str = Field(default='', description='Agent prediction/answer')
    ground_truth: str = Field(default='', description='Ground truth answer')
    env_reward: float = Field(default=0.0, description='Environment reward (e.g., F1, EM)')
    meta_reward: float = Field(default=0.0, description='Computed meta-reward')
    node_outputs: dict[str, Any] = Field(default_factory=dict, description='Per-node outputs in workflow')
    total_llm_calls: int = Field(default=0, ge=0, description='Total LLM API calls made')


class FeedbackCollector:
    '''Collects and persists execution trajectories.

    Stores trajectories as JSONL files at ~/.sema/trajectories/
    '''

    def __init__(self, persistence_dir: str | Path | None = None) -> None:
        if persistence_dir is None:
            from ..config.paths import SEMAPaths
            self._persistence_dir = SEMAPaths.load().trajectories
        else:
            self._persistence_dir = Path(persistence_dir)
        self._trajectories: list[Trajectory] = []

    def record_trajectory(self, trajectory: Trajectory) -> None:
        '''Record a trajectory and persist to disk.'''
        self._trajectories.append(trajectory)
        self._save(trajectory)

    def get_trajectories(self) -> list[Trajectory]:
        '''Return all recorded trajectories.'''
        return list(self._trajectories)

    def _save(self, trajectory: Trajectory) -> None:
        '''Append a trajectory to the JSONL file.'''
        self._persistence_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d')
        filepath = self._persistence_dir / f'trajectories_{timestamp}.jsonl'
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(trajectory.model_dump_json() + '\n')

    def load_trajectories(self, date: str | None = None) -> list[Trajectory]:
        '''Load trajectories from disk.

        Args:
            date: Optional date string (YYYYMMDD) to load specific day's trajectories.
                  If None, loads all.
        '''
        trajectories: list[Trajectory] = []

        if not self._persistence_dir.exists():
            return trajectories

        pattern = f'trajectories_{date}.jsonl' if date else 'trajectories_*.jsonl'
        for filepath in sorted(self._persistence_dir.glob(pattern)):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            data = json.loads(line)
                            trajectories.append(Trajectory(**data))
            except (json.JSONDecodeError, OSError):
                continue

        return trajectories
