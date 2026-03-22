'''HotpotQA Self-Evolving Workflow Example
=========================================

This script demonstrates SEMA's full self-evolution loop on the HotpotQA
multi-hop question-answering benchmark.

What happens at runtime
-----------------------
1. **Seed** — A workflow template (e.g. `chain`) is cloned into a small
   population of ``--population-size`` workflows.
2. **Evaluate** — Each generation, every workflow in the population is run on
   ``--samples`` randomly sampled HotpotQA *validate* examples.  The executor
   fires real LLM calls; a meta-reward scalar is computed from accuracy (F1)
   + efficiency signals.
3. **Optimize** — The chosen optimizer (default: ``hierarchical``) mutates /
   recombines / refines workflow prompts and structure based on the fitness
   scores and recorded trajectories.
4. **Repeat** — Steps 2–3 repeat for ``--generations`` rounds (or until
   ``--early-stop`` consecutive generations show no improvement, or
   ``--fitness-target`` is reached).
5. **Report** — A Markdown summary table and the best-found workflow are
   printed to stdout; full artefacts are persisted under
   ``~/.sema/experiments/<experiment-name>/``.

Quick smoke-test (no LLM, no network):
    python examples/hotpotqa_evolving.py --help

Minimal real run (small, cheap):
    python examples/hotpotqa_evolving.py \\
        --experiment-name hotpotqa_quick \\
        --population-size 2 \\
        --generations 2 \\
        --samples 5

Resume from latest checkpoint:
    python examples/hotpotqa_evolving.py --resume --experiment-name hotpotqa_quick
'''

from __future__ import annotations

import argparse
import asyncio
from typing import Any


# ---------------------------------------------------------------------------
# Lazy import of registry so --help works even if optional deps are missing
# ---------------------------------------------------------------------------
def _optimizer_choices() -> list[str]:
    try:
        from src.optimizer.registry import OptimizerRegistry  # noqa: PLC0415
        return OptimizerRegistry.list_optimizers()
    except Exception:
        return ['hierarchical', 'evolutionary', 'llm', 'rl',
                'self_refinement', 'text_grad', 'cmaes', 'mcts', 'prompt_breeding']


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    '''Parse command-line arguments for the HotpotQA SEMA experiment.'''
    parser = argparse.ArgumentParser(
        description='Run SEMA self-evolving workflow on HotpotQA.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--experiment-name',
        default='hotpotqa_sema_glm_4_plus',
        help='Name of the experiment (used for checkpoint directory).',
    )
    parser.add_argument(
        '--goal',
        default='Answer multi-hop questions accurately by retrieving and reasoning over supporting facts.',
        help='High-level goal passed to all workflows; shapes initial prompts and guides evolution.',
    )
    parser.add_argument(
        '--optimizer',
        default='hierarchical',
        choices=_optimizer_choices(),
        help='Optimizer strategy to drive workflow evolution.',
    )
    parser.add_argument(
        '--template',
        default='auto',
        choices=['auto', 'blank', 'single_agent', 'chain', 'debate', 'hierarchical'],
        help='Seed workflow template to initialise the population.',
    )
    parser.add_argument(
        '--population-size',
        type=int,
        default=4,
        help='Number of workflows in the population.',
    )
    parser.add_argument(
        '--generations',
        type=int,
        default=5,
        help='Maximum number of evolution generations to run.',
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=10,
        help='Number of HotpotQA examples evaluated per workflow per generation.',
    )
    parser.add_argument(
        '--early-stop',
        type=int,
        default=3,
        help='Stop after this many consecutive generations without improvement.',
    )
    parser.add_argument(
        '--fitness-target',
        type=float,
        default=None,
        metavar='FLOAT',
        help='Stop early when best fitness reaches this value (0–1).',
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from the latest (or --checkpoint-gen) checkpoint.',
    )
    parser.add_argument(
        '--checkpoint-gen',
        type=int,
        default=None,
        metavar='INT',
        help='Specific generation checkpoint to resume from (requires --resume).',
    )

    # LLM model overrides
    parser.add_argument(
        '--bootstrap-model',
        default='glm-4-plus',
        help='LLM model used to design the seed workflow when --template=auto.',
    )
    parser.add_argument(
        '--executor-model',
        default='glm-4-plus',
        help='Default LLM model for workflow node agents during execution.',
    )
    parser.add_argument(
        '--optimizer-model',
        default='glm-4-plus',
        help='Default LLM model for all optimizer calls (mutation, crossover, etc.).',
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main async function
# ---------------------------------------------------------------------------

async def main(args: argparse.Namespace) -> None:
    '''Build config, construct orchestrator, run evolution, print results.'''

    from src.workflow.environment import Environment
    from src.orchestrator.config import SEMAConfig
    from src.orchestrator.orchestrator import SEMAOrchestrator

    # ------------------------------------------------------------------
    # 1. Define the task environment
    # ------------------------------------------------------------------
    env = Environment(
        name='hotpotqa',
        benchmark_name='hotpotqa',
        description='Multi-hop question answering — HotpotQA validate split.',
    )

    # ------------------------------------------------------------------
    # 2. Build experiment configuration
    # ------------------------------------------------------------------
    config = SEMAConfig(
        experiment_name=args.experiment_name,
        goal=args.goal,
        environment=env,
        optimizer_type=args.optimizer,
        seed_template=args.template,
        bootstrap_model=args.bootstrap_model,
        executor_model=args.executor_model,
        optimizer_model=args.optimizer_model,
        population_size=args.population_size,
        max_generations=args.generations,
        eval_samples_per_generation=args.samples,
        early_stop_generations=args.early_stop,
        fitness_target=args.fitness_target,
        eval_dataset='validate',
    )

    print(f'Experiment : {config.experiment_name}')
    print(f'Goal       : {config.goal}')
    print(f'Optimizer  : {config.optimizer_type}')
    print(f'Template   : {config.seed_template}')
    print(f'Bootstrap  : {config.bootstrap_model}')
    print(f'Executor   : {config.executor_model}')
    print(f'Opt model  : {config.optimizer_model}')
    print(f'Population : {config.population_size}  workflows')
    print(f'Generations: {config.max_generations}  max')
    print(f'Samples    : {config.eval_samples_per_generation}  per generation')
    print(f'Early-stop : {config.early_stop_generations}  stale generations')
    if config.fitness_target is not None:
        print(f'Target     : {config.fitness_target:.3f}')
    print()

    # ------------------------------------------------------------------
    # 3. Construct orchestrator and run (or resume)
    # ------------------------------------------------------------------
    orch = SEMAOrchestrator(config)

    if args.resume:
        print(f'Resuming from checkpoint (gen={args.checkpoint_gen})...\n')
        result: dict[str, Any] = await orch.resume(args.checkpoint_gen)
    else:
        print('Starting fresh evolution run...\n')
        result = await orch.run()

    # ------------------------------------------------------------------
    # 4. Print generation-by-generation summary table
    # ------------------------------------------------------------------
    history: list[dict[str, Any]] = result.get('history', [])
    if history:
        print('Generation | Best Fitness | Avg Fitness')
        print('-----------|--------------|------------')
        for entry in history:
            gen = entry.get('generation', '?')
            best = entry.get('best_fitness', 0.0)
            avg = entry.get('avg_fitness', 0.0)
            print(f'{gen:>10} | {best:>12.4f} | {avg:>11.4f}')
        print()

    # ------------------------------------------------------------------
    # 5. Print best workflow node names
    # ------------------------------------------------------------------
    best_wf = result.get('best_workflow')
    if best_wf is not None:
        node_names = [node.role.name for node in best_wf.nodes]
        print(f'Best workflow ({best_wf.workflow_id}):')
        print('  Nodes: ' + ' → '.join(node_names) if node_names else '  (no nodes)')
        print(f'  Fitness: {result.get("best_fitness", 0.0):.4f}')
        print()

    # ------------------------------------------------------------------
    # 6. Print full Markdown report
    # ------------------------------------------------------------------
    report: str = result.get('report', '')
    if report:
        print('--- Report ---')
        print(report)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    args = parse_args()
    asyncio.run(main(args))
