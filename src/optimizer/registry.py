'''Optimizer registry for SEMA — mirrors AsyncLLM._provider_registry pattern.'''

from __future__ import annotations

import typing
from typing import ClassVar

from .base import Optimizer, OptimizerConfig
from .population import Population

from .evolutionary import EvolutionaryOptimizer
from .llm_optimizer import LLMOptimizer
from .rl_optimizer import RLOptimizer
from .self_refinement import SelfRefinementOptimizer
from .text_grad import TextGradOptimizer
from .cmaes import CMAESOptimizer
from .mcts import MCTSOptimizer
from .prompt_breeding import PromptBreedingOptimizer
from .hierarchical import HierarchicalOptimizer


class OptimizerRegistry:
    '''Registry for optimizer implementations.

    Usage:
        OptimizerRegistry.register("my_opt", MyOptimizer)
        opt = OptimizerRegistry.create("my_opt", config, population)
    '''

    _registry: ClassVar[dict[str, type[Optimizer]]] = {}

    @classmethod
    def register(cls, name: str, optimizer_class: type[Optimizer]) -> None:
        '''Register an optimizer class under a name.'''
        cls._registry[name] = optimizer_class

    @classmethod
    def create(cls, name: str, config: OptimizerConfig, population: Population) -> Optimizer:
        '''Instantiate a registered optimizer by name.

        Raises:
            KeyError: if name is not registered.
        '''
        if name not in cls._registry:
            available = ', '.join(cls._registry.keys())
            raise KeyError(f"Optimizer '{name}' not found. Available: {available}")
        optimizer_class = cls._registry[name]
        # Coerce base OptimizerConfig to the specific subclass expected by this optimizer
        try:
            hints = typing.get_type_hints(optimizer_class.__init__)
            config_type = hints.get('config')
            if (
                config_type is not None
                and isinstance(config_type, type)
                and issubclass(config_type, OptimizerConfig)
                and not isinstance(config, config_type)
            ):
                config = config_type(**config.model_dump())
        except Exception:
            pass
        return optimizer_class(config, population)

    @classmethod
    def list_optimizers(cls) -> list[str]:
        '''Return list of registered optimizer names.'''
        return list(cls._registry.keys())


# ---------------------------------------------------------------------------
# Default registrations
# ---------------------------------------------------------------------------
OptimizerRegistry.register('evolutionary', EvolutionaryOptimizer)
OptimizerRegistry.register('llm', LLMOptimizer)
OptimizerRegistry.register('rl', RLOptimizer)
OptimizerRegistry.register('self_refinement', SelfRefinementOptimizer)
OptimizerRegistry.register('text_grad', TextGradOptimizer)
OptimizerRegistry.register('cmaes', CMAESOptimizer)
OptimizerRegistry.register('mcts', MCTSOptimizer)
OptimizerRegistry.register('prompt_breeding', PromptBreedingOptimizer)
OptimizerRegistry.register('hierarchical', HierarchicalOptimizer)
