'''Pure-Python CMA-ES optimizer for continuous workflow hyperparameters.'''

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from pydantic import Field

from ..workflow.schema import Workflow
from ..feedback.base import Trajectory
from .base import Optimizer, OptimizerConfig
from .population import Population


# ---------------------------------------------------------------------------
# Pure-Python linear algebra helpers
# ---------------------------------------------------------------------------

def _mat_vec(M: list[list[float]], v: list[float]) -> list[float]:
    '''Matrix-vector product M @ v.'''
    n = len(M)
    return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(n)]


def _vec_add(a: list[float], b: list[float]) -> list[float]:
    return [x + y for x, y in zip(a, b)]


def _vec_scale(v: list[float], s: float) -> list[float]:
    return [x * s for x in v]


def _vec_sub(a: list[float], b: list[float]) -> list[float]:
    return [x - y for x, y in zip(a, b)]


def _outer(a: list[float], b: list[float]) -> list[list[float]]:
    '''Outer product a ⊗ b.'''
    return [[ai * bj for bj in b] for ai in a]


def _mat_scale(M: list[list[float]], s: float) -> list[list[float]]:
    return [[M[i][j] * s for j in range(len(M[i]))] for i in range(len(M))]


def _mat_add(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    n = len(A)
    return [[A[i][j] + B[i][j] for j in range(len(A[i]))] for i in range(n)]


def _identity(n: int) -> list[list[float]]:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _cholesky(A: list[list[float]]) -> list[list[float]]:
    '''Lower Cholesky decomposition: A = L L^T (pure Python).'''
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                val = A[i][i] - s
                L[i][j] = math.sqrt(max(val, 1e-12))
            else:
                L[i][j] = (A[i][j] - s) / L[j][j] if L[j][j] > 1e-12 else 0.0
    return L


def _sample_normal(mean: list[float], sigma: float, L: list[list[float]]) -> list[float]:
    '''Sample x ~ N(mean, sigma^2 * L L^T).'''
    n = len(mean)
    z = [random.gauss(0, 1) for _ in range(n)]
    Lz = _mat_vec(L, z)
    return _vec_add(mean, _vec_scale(Lz, sigma))


def _condition_number(C: list[list[float]]) -> float:
    '''Approximate condition number via diagonal (rough estimate).'''
    diag = [C[i][i] for i in range(len(C))]
    mx = max(diag) if diag else 1.0
    mn = min(d for d in diag if d > 0) if any(d > 0 for d in diag) else 1.0
    return mx / mn


# ---------------------------------------------------------------------------
# CMA-ES state
# ---------------------------------------------------------------------------

@dataclass
class _CMAESState:
    '''Internal CMA-ES distribution state.'''

    dim: int
    mean: list[float] = field(default_factory=list)
    sigma: float = 0.5
    C: list[list[float]] = field(default_factory=list)   # covariance matrix
    pc: list[float] = field(default_factory=list)         # evolution path for C
    ps: list[float] = field(default_factory=list)         # evolution path for sigma
    generation: int = 0

    def __post_init__(self) -> None:
        if not self.mean:
            self.mean = [0.0] * self.dim
        if not self.C:
            self.C = _identity(self.dim)
        if not self.pc:
            self.pc = [0.0] * self.dim
        if not self.ps:
            self.ps = [0.0] * self.dim


# ---------------------------------------------------------------------------
# Config + Optimizer
# ---------------------------------------------------------------------------

class CMAESConfig(OptimizerConfig):
    '''Configuration for CMA-ES optimizer.'''

    sigma0: float = Field(default=0.5, gt=0.0)
    target_params: list[str] = Field(default_factory=lambda: ['temperature', 'max_steps'])


class CMAESOptimizer(Optimizer):
    '''CMA-ES over continuous workflow hyperparameters (pure Python, no numpy).'''

    def __init__(self, config: CMAESConfig, population: Population) -> None:
        super().__init__(config, population)
        self.config: CMAESConfig = config
        self._state: _CMAESState | None = None
        self._stats: dict[str, Any] = {
            'generations': 0,
            'sigma': config.sigma0,
            'mean': [],
            'condition_number': 1.0,
        }

    # -----------------------------------------------------------------------
    # Parameter extraction / injection
    # -----------------------------------------------------------------------

    def _extract_params(self, workflow: Workflow) -> list[float]:
        params: list[float] = []
        for node in workflow.nodes:
            for pname in self.config.target_params:
                if pname == 'temperature':
                    params.append(float(node.agent_config.get('temperature', 0.7)))
                elif pname == 'max_steps':
                    params.append(float(node.agent_config.get('max_steps', node.action.max_steps)))
                else:
                    params.append(float(node.agent_config.get(pname, 0.0)))
        return params

    def _inject_params(self, workflow: Workflow, params: list[float]) -> Workflow:
        idx = 0
        child = self._deep_copy_workflow(workflow)
        n_params = len(self.config.target_params)
        for node in child.nodes:
            cfg = dict(node.agent_config)
            for pname in self.config.target_params:
                if idx >= len(params):
                    break
                val = params[idx]
                if pname == 'temperature':
                    cfg['temperature'] = max(0.0, min(2.0, val))
                elif pname == 'max_steps':
                    cfg['max_steps'] = max(1, min(10, int(round(val))))
                else:
                    cfg[pname] = val
                idx += 1
            node.agent_config = cfg
        return child

    # -----------------------------------------------------------------------
    # CMA-ES update
    # -----------------------------------------------------------------------

    def _cmaes_update(
        self,
        state: _CMAESState,
        samples: list[list[float]],
        fitnesses: list[float],
        mu: int,
        lam: int,
    ) -> _CMAESState:
        '''One CMA-ES update step. Mutates state in place.'''
        n = state.dim

        # Sort by fitness (descending); zip truncates to min length
        ranked = sorted(zip(fitnesses, samples), key=lambda x: -x[0])
        mu = min(mu, len(ranked))
        if mu == 0:
            return state
        elite_samples = [s for _, s in ranked[:mu]]

        # CMA-ES hyperparameters (standard settings, computed with actual mu)
        weights = [math.log(mu + 0.5) - math.log(i + 1) for i in range(mu)]
        w_sum = sum(weights)
        weights = [w / w_sum for w in weights]
        mu_eff = 1.0 / sum(w ** 2 for w in weights)

        cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        cs = (mu_eff + 2) / (n + mu_eff + 5)
        c1 = 2.0 / ((n + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        damps = 1 + 2 * max(0, math.sqrt((mu_eff - 1) / (n + 1)) - 1) + cs
        chi_n = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # New mean
        old_mean = state.mean[:]
        new_mean = [sum(weights[i] * elite_samples[i][j] for i in range(mu)) for j in range(n)]
        state.mean = new_mean

        # Sigma path (ps)
        invsqrt_C = _identity(n)  # Approximation: identity (simplified)
        mean_diff = _vec_scale(_vec_sub(new_mean, old_mean), 1.0 / state.sigma)
        invsqrt_diff = _mat_vec(invsqrt_C, mean_diff)

        state.ps = _vec_add(
            _vec_scale(state.ps, 1 - cs),
            _vec_scale(invsqrt_diff, math.sqrt(cs * (2 - cs) * mu_eff)),
        )

        hs = (
            math.sqrt(sum(x ** 2 for x in state.ps)) / math.sqrt(1 - (1 - cs) ** (2 * (state.generation + 1)))
            < (1.4 + 2 / (n + 1)) * chi_n
        )

        # Covariance evolution path (pc)
        state.pc = _vec_add(
            _vec_scale(state.pc, 1 - cc),
            _vec_scale(mean_diff, (1 if hs else 0) * math.sqrt(cc * (2 - cc) * mu_eff)),
        )

        # Covariance matrix update
        rank_one = _mat_scale(_outer(state.pc, state.pc), c1)
        rank_mu_sum = [[0.0] * n for _ in range(n)]
        for i in range(mu):
            di = _vec_scale(_vec_sub(elite_samples[i], old_mean), 1.0 / state.sigma)
            rank_mu_sum = _mat_add(rank_mu_sum, _mat_scale(_outer(di, di), weights[i]))
        rank_mu = _mat_scale(rank_mu_sum, cmu)

        state.C = _mat_add(
            _mat_scale(state.C, 1 - c1 - cmu),
            _mat_add(rank_one, rank_mu),
        )

        # Sigma update
        ps_norm = math.sqrt(sum(x ** 2 for x in state.ps))
        state.sigma *= math.exp((cs / damps) * (ps_norm / chi_n - 1))
        state.sigma = max(1e-6, min(10.0, state.sigma))

        state.generation += 1
        return state

    # -----------------------------------------------------------------------
    # step
    # -----------------------------------------------------------------------

    async def step(
        self,
        population: Population,
        fitness_scores: dict[str, float],
        trajectories: list[Trajectory],
    ) -> Population:
        for wf_id, score in fitness_scores.items():
            population.update_fitness(wf_id, score)

        elite = self._get_elite(population)
        if not elite or not elite[0].nodes:
            population.advance_generation()
            return population

        # Determine dimensionality
        dim = len(self._extract_params(elite[0]))
        if dim == 0:
            population.advance_generation()
            return population

        # Initialize state on first call
        if self._state is None or self._state.dim != dim:
            init_mean = self._extract_params(elite[0])
            self._state = _CMAESState(dim=dim, mean=init_mean, sigma=self.config.sigma0)

        state = self._state
        lam = self.config.population_size
        mu = max(1, lam // 2)

        # Compute Cholesky factor
        try:
            L = _cholesky(state.C)
        except Exception:
            state.C = _identity(dim)
            L = _cholesky(state.C)

        # Sample λ new parameter vectors
        samples = [_sample_normal(state.mean, state.sigma, L) for _ in range(lam)]

        # Evaluate fitness: use existing workflow fitness as proxy
        all_workflows = population.workflows
        wf_fitnesses = [wf.fitness for wf in all_workflows]

        # CMA-ES update
        self._state = self._cmaes_update(state, samples, wf_fitnesses, mu, lam)

        # Inject new parameters into workflows
        new_workflows: list[Workflow] = [self._deep_copy_workflow(wf) for wf in elite]
        n_new = self.config.population_size - len(elite)

        template = elite[0]
        for i in range(n_new):
            child = self._inject_params(template, samples[i % len(samples)])
            child.parent_ids = [template.workflow_id]
            new_workflows.append(child)

        new_workflows = new_workflows[: self.config.population_size]
        population.replace_workflows(new_workflows)
        population.advance_generation()

        self._stats['generations'] += 1
        self._stats['sigma'] = self._state.sigma
        self._stats['mean'] = list(self._state.mean)
        self._stats['condition_number'] = _condition_number(self._state.C)

        return population

    def get_statistics(self) -> dict[str, Any]:
        return dict(self._stats)
