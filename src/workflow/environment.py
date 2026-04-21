'''Environment model for SEMA workflows.'''

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, ConfigDict


class Environment(BaseModel):
    '''Describes the environment/benchmark context for a workflow.'''

    model_config = ConfigDict(validate_assignment=True)

    name: str = Field(description='Environment name')
    benchmark_name: str = Field(default='', description='Name of the benchmark to use')
    dataset: str = Field(default='', description='Dataset identifier or path')
    description: str = Field(default='', description='Environment description')
    constraints: dict[str, Any] = Field(default_factory=dict, description='Environment constraints')
    metadata: dict[str, Any] = Field(default_factory=dict, description='Arbitrary metadata')

    _previous_state: dict[str, Any] | None = None

    def get_benchmark(self):
        '''Instantiate and return the benchmark from benchmark_name.'''
        from ..benchmarks import HotpotQA

        benchmarks = {
            'hotpotqa': HotpotQA,
        }

        name_lower = self.benchmark_name.lower()
        if name_lower not in benchmarks:
            raise ValueError(f'Unknown benchmark: {self.benchmark_name}. Available: {list(benchmarks.keys())}')

        return benchmarks[name_lower]()

    def has_changed(self, previous: Environment) -> bool:
        '''Check if this environment differs from a previous one.'''
        return (
            self.name != previous.name
            or self.benchmark_name != previous.benchmark_name
            or self.dataset != previous.dataset
            or self.constraints != previous.constraints
        )
