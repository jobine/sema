# SEMA (Self-Evolving Multi-Agent) 实现计划

> SEMA introduces a lifelong, self-evolving loop where a population of agents continually refines their prompts, memory, tool-use strategies and even their interaction patterns based on environmental feedback and meta-rewards.

## 一、整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                       SEMAOrchestrator                              │
│         (self-evolving main loop, lifecycle, experiment tracking)    │
├───────────┬───────────┬────────────┬────────────┬──────────────────┤
│Optimization│  Workflow  │ Feedback & │   Memory   │     Tool         │
│  Engine    │  Engine    │ Meta-Reward│   System   │     System       │
│            │           │            │            │                  │
│ - Optimizer│ - Schema  │ - Env.     │ - Short    │ - Registry       │
│   Registry │ - Executor│   feedback │   term     │ - Execution      │
│ - EA / LLM │ - Template│ - Meta-    │ - Long     │ - Strategy       │
│ / RL / SR  │ - Environ.│   reward   │   term     │   evolution      │
│ / TextGrad │           │ - Shaping  │ - Retrieval│                  │
├───────────┴───────────┴────────────┴────────────┴──────────────────┤
│           Workflow (executable DAG + evolvable genome)              │
│   Role/Action per node, JSON-serializable, topology-evolvable      │
├─────────────────────────────────────────────────────────────────────┤
│                 Agent (extended with evolution support)              │
│     (evolvable prompts, memory access, tool use)                    │
├─────────────────────────────────────────────────────────────────────┤
│                    Existing SEMA Infrastructure                   │
│          AgentConfig / AgentState / AsyncLLM / Benchmark            │
└─────────────────────────────────────────────────────────────────────┘
```

## 二、分阶段实现计划

### 阶段 1：可演化智能体与记忆系统

**目标**：直接扩展现有 Agent/AgentConfig/AgentState，增加演化支持和双层记忆系统。

#### 1.1 扩展现有 Agent 类

**文件**: `src/agents/agent.py`（修改现有文件）

对 `AgentConfig`、`AgentState`、`Agent` 三个类进行就地修改，不创建子类。

```python
class AgentConfig(BaseModel):
    """Configuration for an agent.

    The 4 core fields (model, max_steps, verbose, temperature) remain fixed.
    Evolution-related fields (system_prompt, reasoning_prompt, answer_prompt,
    agent_id, generation, parent_ids, max_reasoning_depth, tool_use_strategy,
    memory_strategy) are passed as extra kwargs only when used in evolution
    context — no need to declare them as formal fields.
    """

    model_config = ConfigDict(
        extra='allow',           # <-- changed from 'forbid' to 'allow'
        validate_assignment=True,
    )

    # --- existing 4 fields, unchanged ---
    model: str = Field(default='gpt-4o-mini', description='LLM model name to use')
    max_steps: int = Field(default=5, ge=1, description='Maximum number of reasoning steps')
    verbose: bool = Field(default=False, description='Whether to print verbose output')
    temperature: float | None = Field(default=None, ge=0.0, le=2.0, description='LLM temperature')


class AgentState(BaseModel):
    """State of an agent during execution.

    Already has extra='allow'. Added trajectory tracking fields for evolution.
    """

    model_config = ConfigDict(
        extra='allow',
        validate_assignment=True,
    )

    # --- existing fields ---
    question: str = Field(default='', description='The question being answered')
    context: str = Field(default='', description='Context information for the question')
    steps: list[dict[str, Any]] = Field(default_factory=list, description='History of reasoning steps')
    answer: str = Field(default='', description='The current answer')
    finished: bool = Field(default=False, description='Whether the agent has finished')

    # --- new trajectory tracking fields ---
    memory_retrievals: list[dict] = Field(default_factory=list, description='Memory retrieval records')
    tool_calls: list[dict] = Field(default_factory=list, description='Tool call records')
    reasoning_trace: list[str] = Field(default_factory=list, description='Reasoning chain trace')
    reward: float = Field(default=0.0, description='Reward for current execution')
    metadata: dict = Field(default_factory=dict, description='Extension metadata')


class Agent(BaseModel, ABC):
    """Abstract base class for all agents.

    Extended with evolution support: genome export/import, trajectory capture,
    prompt building, memory and tool access.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )

    config: AgentConfig = Field(default_factory=AgentConfig)
    _llm: AsyncBaseLLM | None = PrivateAttr(default=None)
    _memory: MemorySystem | None = PrivateAttr(default=None)    # lazy init
    _tools: ToolRegistry | None = PrivateAttr(default=None)     # lazy init
    _state: AgentState | None = PrivateAttr(default=None)

    @property
    def llm(self) -> AsyncBaseLLM:
        """Lazy initialization of LLM."""
        if self._llm is None:
            self._llm = AsyncLLM(self.config.model)
        return self._llm

    @abstractmethod
    async def run(self, question: str, **kwargs: Any) -> str:
        """Run the agent on a given question and return the answer."""
        ...

    @abstractmethod
    async def step(self, state: AgentState) -> AgentState:
        """Execute one step of the agent's reasoning."""
        ...

    def build_prompt(self, state: AgentState) -> str:
        """Build prompt from templates + memory + tool list.

        Uses system_prompt, reasoning_prompt, answer_prompt from
        config extras if present, otherwise uses defaults.
        """
        ...

    def get_genome(self) -> dict:
        """Export evolvable parameters as a 'genome' dict.

        Includes: all config fields (core + extras), prompt templates,
        strategy parameters, lineage (agent_id, generation, parent_ids).
        """
        ...

    @classmethod
    def from_genome(cls, genome: dict) -> Agent:
        """Create an agent instance from a genome dict.

        Reconstructs AgentConfig from genome and instantiates
        the appropriate Agent subclass.
        """
        ...

    def get_trajectory(self) -> dict:
        """Export the complete execution trajectory for feedback and evolution."""
        ...

    def reset(self) -> None:
        """Reset the agent's state for a new question."""
        self._llm = None
        self._state = None
```

**设计要点**：
- 直接修改现有 `Agent`，不创建子类，保持继承链简洁
- `AgentConfig` 改为 `extra='allow'`，演化参数作为 extra kwargs 传入，无需声明正式字段
- `AgentState` 已有 `extra='allow'`，新增 trajectory tracking 字段
- `_memory`、`_tools`、`_state` 作为 `PrivateAttr` 惰性初始化，与现有 `_llm` 模式一致
- `get_genome()` / `from_genome()` 实现基因组与运行时配置的双向转换

#### 1.2 记忆系统

**文件**: `src/memory/base.py`, `src/memory/short_term.py`, `src/memory/long_term.py`, `src/memory/memory_system.py`

```python
# --- base.py ---
class MemoryEntry(BaseModel):
    entry_id: str               # uuid
    content: str                # memory content
    entry_type: str             # "observation" | "reasoning" | "feedback" | "tool_result"
    timestamp: float
    importance: float           # 0.0 ~ 1.0
    metadata: dict


class MemoryStore(ABC):
    async def add(self, entry: MemoryEntry) -> None
    async def retrieve(self, query: str, top_k: int = 5) -> list[MemoryEntry]
    async def clear(self) -> None
    def size(self) -> int


# --- short_term.py ---
class ShortTermMemory(MemoryStore):
    """Fixed-capacity sliding window storing current task execution context."""
    # FIFO eviction policy
    # Recency-based retrieval


# --- long_term.py ---
class LongTermMemory(MemoryStore):
    """Persistent cross-task/cross-generation memory."""
    # Weighted retrieval: similarity + importance + time decay
    # Persisted as JSONL to ~/.sema/memory/{agent_id}/
    # Initial implementation uses TF-IDF cosine similarity (no vector DB dependency)


# --- memory_system.py ---
class MemorySystem:
    """Unified memory interface integrating short-term and long-term memory."""
    async def remember(self, content: str, entry_type: str = "observation") -> None
    async def recall(self, query: str, top_k: int = 5, source: str = "both") -> list[MemoryEntry]
    async def consolidate(self) -> None       # migrate important short-term entries to long-term
    async def summarize_context(self, query: str, max_tokens: int = 500) -> str
    def reset_short_term(self) -> None        # reset only short-term memory (between tasks)
```

---

### 阶段 2：工具系统与反馈/元奖励

**目标**：建立工具注册执行框架，以及环境反馈收集和元奖励计算能力。

#### 2.1 工具系统

**文件**: `src/tools/base.py`, `src/tools/registry.py`, `src/tools/builtin/`

```python
# --- base.py ---
class ToolSpec(BaseModel):
    """Tool specification (for LLM understanding)."""
    name: str
    description: str
    parameters: list[ToolParameter]
    returns: str


class ToolResult(BaseModel):
    """Tool execution result."""
    tool_name: str
    success: bool
    output: Any
    error: str | None
    execution_time: float


class Tool(ABC):
    @property
    @abstractmethod
    def spec(self) -> ToolSpec
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult


# --- registry.py ---
class ToolRegistry:
    """Tool registry using the same ClassVar registration pattern as AsyncLLM."""
    _global_registry: ClassVar[dict[str, Type[Tool]]] = {}

    def register(self, tool: Tool) -> None
    def get(self, name: str) -> Tool | None
    def list_tools(self) -> list[ToolSpec]
    async def execute(self, tool_name: str, **kwargs) -> ToolResult
    def format_for_prompt(self) -> str      # format tool list as prompt text
    @classmethod
    def register_global(cls, name: str, tool_class: Type[Tool]) -> None


# --- builtin/ ---
class SearchTool(Tool)        # search within context
class CalculatorTool(Tool)    # mathematical calculations
class LookupTool(Tool)        # lookup from memory
```

#### 2.2 反馈与元奖励

**文件**: `src/feedback/base.py`, `src/feedback/meta_reward.py`, `src/feedback/reward_shaping.py`

```python
# --- base.py ---
class Trajectory(BaseModel):
    """Complete trajectory of a single task execution."""
    workflow_id: str
    question: str
    context: str
    steps: list[dict]
    memory_retrievals: list[dict]
    tool_calls: list[dict]
    prediction: str
    ground_truth: str
    env_reward: float           # EM/F1 from benchmark
    meta_reward: float          # meta-reward
    node_outputs: dict = {}     # per-node outputs from workflow execution
    total_llm_calls: int = 0


class FeedbackCollector:
    """Collect and store trajectory data."""
    async def record_trajectory(self, trajectory: Trajectory) -> None
    async def get_trajectories(self, workflow_id: str = None, generation: int = None) -> list[Trajectory]
    # Persistence: ~/.sema/trajectories/{experiment_id}/ (JSONL)


# --- meta_reward.py ---
class MetaRewardConfig(BaseModel):
    accuracy_weight: float = 0.5        # accuracy weight
    efficiency_weight: float = 0.2      # efficiency (inverse of step count)
    tool_use_weight: float = 0.15       # tool use quality
    memory_use_weight: float = 0.15     # memory utilization quality


class MetaRewardComputer:
    """Synthesize multi-dimensional feedback signals into scalar fitness."""
    def compute(self, trajectory: Trajectory) -> float
    def compute_batch(self, trajectories: list[Trajectory]) -> list[float]


# --- reward_shaping.py ---
class RewardShaper:
    """Reward shaping: normalization / baseline removal / ranking."""
    def shape(self, rewards: list[float]) -> list[float]
    def fitness_from_rewards(self, rewards: list[float]) -> list[float]
```

**与 Benchmark 的集成**：
- `Benchmark.evaluate()` 返回 `{'em', 'f1', 'result'}`，其中 `f1` 直接作为 `Trajectory.env_reward`
- `MetaRewardComputer` 在评估后结合轨迹的效率、工具使用、记忆利用等维度计算综合元奖励
- 评估阶段直接遍历数据集并复用 `Benchmark.evaluate()` 计算指标，而非使用 `Benchmark.run()`，以便收集更丰富的轨迹信息

---

### 阶段 3：优化引擎（抽象化多策略优化）

**目标**：实现抽象优化器框架，支持多种优化策略（演化算法、LLM 优化、强化学习、自我改进、文本梯度）。

> 参考：[arxiv 2508.07407](https://arxiv.org/abs/2508.07407) Section 4 & 5.2

#### 3.1 种群管理（共享基础设施）

**文件**: `src/optimizer/population.py`

`AgentGenome` is removed. The evolvable unit is now `Workflow` (defined in `src/workflow/schema.py`). Each `Workflow` contains `WorkflowNode`s with `Role`, `Action`, and `agent_config`, so all evolvable parameters are captured within the workflow graph structure. Fitness is tracked directly on the `Workflow` object (`fitness`, `fitness_history`).

```python
from src.workflow.schema import Workflow


class Population:
    """Population manager shared across optimizers.

    Holds a list of Workflows as the evolvable units.
    """
    def initialize(self, seed_workflows: list[Workflow] | None = None) -> None
    @property
    def workflows(self) -> list[Workflow]
    @property
    def best_workflow(self) -> Workflow
    def get_elite(self, n: int = None) -> list[Workflow]
    def update_fitness(self, workflow_id: str, fitness: float) -> None
    def advance_generation(self) -> None
    async def save(self, path: Path) -> None
    @classmethod
    async def load(cls, path: Path) -> Population
    def summary(self) -> dict         # avg fitness, best, worst, diversity
```

#### 3.2 抽象优化器接口

**文件**: `src/optimizer/base.py`

```python
class OptimizerConfig(BaseModel):
    """Base configuration for all optimizers."""
    population_size: int = 10
    max_generations: int = 50
    elitism_rate: float = 0.2


class Optimizer(ABC):
    """Abstract optimizer interface.

    All optimization strategies (evolutionary, LLM-based, RL, self-refinement,
    text-gradient) implement this unified interface. Each optimizer takes the
    current population, fitness scores, and execution trajectories, and produces
    an updated population.
    """

    def __init__(self, config: OptimizerConfig, population: Population):
        self.config = config
        self.population = population

    @abstractmethod
    async def step(
        self,
        population: Population,
        fitness_scores: dict[str, float],
        trajectories: list[Trajectory],
    ) -> Population:
        """Run one optimization step and return updated population.

        Args:
            population: Current workflow population.
            fitness_scores: Map of workflow_id -> scalar fitness.
            trajectories: Execution trajectories from the evaluation phase.

        Returns:
            Updated population after optimization.
        """
        ...

    @abstractmethod
    def get_statistics(self) -> dict:
        """Return optimizer-specific statistics for logging."""
        ...
```

#### 3.3 优化器实现

| 优化器 | 机制 | 关键方法 | 文件 |
|--------|------|---------|------|
| `EvolutionaryOptimizer` | GA: population, selection, crossover, mutation at 3 levels | `step()` via `evolve_generation()` | `src/optimizer/evolutionary.py` |
| `LLMOptimizer` | LLM-as-Optimizer (OPRO-style): LLM generates improved workflows from trajectory feedback | `step()` via `optimize()` | `src/optimizer/llm_optimizer.py` |
| `RLOptimizer` | RL-based: reward-guided policy gradient on prompts/strategies | `step()` via `update_policy()` | `src/optimizer/rl_optimizer.py` |
| `SelfRefinementOptimizer` | Iterative self-reflection: workflow critiques own trajectories and improves | `step()` via `refine()` | `src/optimizer/self_refinement.py` |
| `TextGradOptimizer` | Semantic gradient: textual feedback as "gradients" for prompt updates | `step()` via `backpropagate()` | `src/optimizer/text_grad.py` |
| `CMAESOptimizer` | CMA-ES: 协方差矩阵自适应进化策略，高效优化连续/离散超参数 | `step()` via `sample_and_update()` | `src/optimizer/cmaes.py` |
| `MCTSOptimizer` | MCTS: 蒙特卡洛树搜索，结构化探索 DAG 拓扑空间 | `step()` via `search()` | `src/optimizer/mcts.py` |
| `PromptBreedingOptimizer` | LLM 驱动的语义级 prompt 交叉与变异 | `step()` via `breed()` | `src/optimizer/prompt_breeding.py` |
| **`HierarchicalOptimizer`** | **推荐默认：分层混合优化器，按层级调度最优子优化器** | `step()` dispatches to sub-optimizers | `src/optimizer/hierarchical.py` |

**Mutation operates at 3 levels**:
- **Micro**: mutate `agent_config` within nodes (model, temperature, max_steps)
- **Meso**: mutate Role/Action on nodes (system_prompt, instruction_prompt, tools, strategy) — LLM-based
- **Macro**: mutate topology (add/remove nodes, add/remove edges, rewire)

**文件**: `src/optimizer/evolutionary.py`

```python
from src.workflow.schema import Workflow

# --- Mutation operators (3 levels) ---
class MutationOperator(ABC):
    def mutate(self, workflow: Workflow, mutation_rate: float = 0.1) -> Workflow

class NodeConfigMutator(MutationOperator):
    """Micro-level: Gaussian noise for numeric params (temperature, max_steps),
    random switch for discrete params (model) within workflow nodes."""

class RoleActionMutator(MutationOperator):
    """Meso-level: Use LLM to rewrite/enhance Role.system_prompt,
    Action.instruction_prompt, available_tools, memory_strategy."""
    async def _mutate_with_llm(self, text: str, instruction: str) -> str

class TopologyMutator(MutationOperator):
    """Macro-level: Add/remove nodes, add/remove edges, rewire connections.
    Ensures DAG validity after mutation."""
    def _add_node(self, workflow: Workflow) -> Workflow
    def _remove_node(self, workflow: Workflow) -> Workflow
    def _rewire_edge(self, workflow: Workflow) -> Workflow

# --- Crossover ---
class CrossoverOperator(ABC):
    def crossover(self, parent1: Workflow, parent2: Workflow) -> Workflow

class WorkflowCrossover(CrossoverOperator):
    """Merge topologies from two parent workflows.
    Selects subgraphs from each parent and combines them."""

# --- Selection (unchanged, operates on fitness) ---
class SelectionOperator(ABC):
    def select(self, workflows: list[Workflow], n: int) -> list[Workflow]

class TournamentSelection(SelectionOperator):
    """Tournament selection."""

class FitnessProportionalSelection(SelectionOperator):
    """Fitness-proportional (roulette wheel) selection."""


class EvolutionaryConfig(OptimizerConfig):
    mutation_rate: float = 0.1
    crossover_rate: float = 0.3
    selection_method: str = "tournament"
    mutation_levels: list[str] = ["micro", "meso", "macro"]
    macro_mutation_rate: float = 0.05   # lower rate for topology changes


class EvolutionaryOptimizer(Optimizer):
    """Genetic algorithm optimizer: elitism -> selection -> crossover -> mutation.

    Mutation is applied at micro/meso/macro levels with configurable rates.
    """

    async def step(self, population, fitness_scores, trajectories) -> Population:
        # elite preservation -> parent selection -> crossover -> mutation -> new population
        ...

    def get_statistics(self) -> dict:
        ...
```

**文件**: `src/optimizer/llm_optimizer.py`

```python
class LLMOptimizerConfig(OptimizerConfig):
    meta_llm_model: str = "gpt-4o"      # LLM used for optimization
    num_candidates: int = 5              # candidates generated per step
    context_window: int = 20             # number of past trajectories shown to LLM


class LLMOptimizer(Optimizer):
    """OPRO-style optimizer: LLM generates improved workflows from trajectory feedback.

    Reference: Yang et al., "Large Language Models as Optimizers" (2023)
    The meta-LLM sees past (workflow, score) pairs and proposes improved workflows.
    Can also bootstrap blank workflows by analyzing the goal and proposing initial structure.
    """

    async def step(self, population, fitness_scores, trajectories) -> Population:
        # Build meta-prompt with top-k past (workflow, fitness) pairs
        # Ask meta-LLM to propose improved workflows (roles, actions, topology)
        # For blank workflows: analyze goal and propose initial node structure
        # Evaluate and retain best
        ...

    def get_statistics(self) -> dict:
        ...
```

**文件**: `src/optimizer/rl_optimizer.py`

```python
class RLOptimizerConfig(OptimizerConfig):
    learning_rate: float = 0.01
    discount_factor: float = 0.99
    policy_type: str = "prompt_gradient"   # "prompt_gradient" | "bandit"


class RLOptimizer(Optimizer):
    """RL-based optimizer: reward-guided policy gradient on prompts/strategies.

    Treats prompt/strategy selection as a policy optimization problem.
    Uses reward signals from trajectories to update the policy.
    """

    async def step(self, population, fitness_scores, trajectories) -> Population:
        # Compute advantages from trajectory rewards
        # Update policy parameters using policy gradient
        # Generate new population from updated policy
        ...

    def get_statistics(self) -> dict:
        ...
```

**文件**: `src/optimizer/self_refinement.py`

```python
class SelfRefinementConfig(OptimizerConfig):
    max_reflection_rounds: int = 3
    critique_model: str = "gpt-4o"


class SelfRefinementOptimizer(Optimizer):
    """Iterative self-reflection: agent critiques own trajectories and improves.

    Each agent reviews its own failures, generates a self-critique,
    and proposes concrete improvements to its prompts/strategies.
    """

    async def step(self, population, fitness_scores, trajectories) -> Population:
        # For each agent: gather failed trajectories
        # Generate self-critique via LLM
        # Propose and apply improvements
        ...

    def get_statistics(self) -> dict:
        ...
```

**文件**: `src/optimizer/text_grad.py`

```python
class TextGradConfig(OptimizerConfig):
    gradient_model: str = "gpt-4o"
    step_size: str = "medium"        # "small" | "medium" | "large"


class TextGradOptimizer(Optimizer):
    """Semantic gradient optimizer: textual feedback as 'gradients' for prompt updates.

    Reference: Yuksekgonul et al., "TextGrad" (2024)
    Computes textual 'gradients' by comparing predictions to ground truth,
    then applies these gradients to update prompts.
    """

    async def step(self, population, fitness_scores, trajectories) -> Population:
        # Compute textual gradients from trajectory errors
        # Apply gradient-based prompt updates
        # Return updated population
        ...

    def get_statistics(self) -> dict:
        ...
```

#### 3.4 新增优化器实现

**文件**: `src/optimizer/cmaes.py`

```python
class CMAESConfig(OptimizerConfig):
    sigma0: float = 0.5              # initial step size
    target_params: list[str] = ["temperature", "max_steps"]  # params to optimize


class CMAESOptimizer(Optimizer):
    """CMA-ES optimizer for continuous workflow hyperparameters.

    Maintains a multivariate normal distribution over numeric params
    (temperature, max_steps, etc.). Self-adapts step size and search
    direction based on fitness ranking. Much higher sample efficiency
    than GA for continuous parameter tuning.

    Population size is auto-determined: 4 + floor(3 * ln(dim)).
    Does NOT handle prompt or topology optimization — delegate those
    to meso/macro sub-optimizers.

    Dependency: cmaes (pip install cmaes)
    """

    async def step(self, population, fitness_scores, trajectories) -> Population:
        # Extract numeric params from each workflow's nodes → vector
        # Feed (vectors, fitness) to CMA-ES
        # Sample new vectors from updated distribution
        # Write back to workflow node configs
        ...

    def get_statistics(self) -> dict:
        # sigma, mean, condition number
        ...
```

**文件**: `src/optimizer/mcts.py`

```python
class MCTSConfig(OptimizerConfig):
    exploration_constant: float = 1.41   # UCB1 C parameter
    max_iterations: int = 20             # MCTS iterations per step
    rollout_samples: int = 3             # eval samples per rollout
    max_depth: int = 6                   # max topology decisions per rollout


class MCTSOptimizer(Optimizer):
    """Monte Carlo Tree Search for workflow topology optimization.

    Models workflow construction as a sequential decision tree:
      - Level 0: choose number of nodes (1-5)
      - Level 1..N: for each node, choose role type
      - Level N+1..M: choose edge connections
      - Level M+1..K: choose edge types (data/conditional/loop)

    Each node in the search tree = one topology decision.
    Uses UCB1 for exploration-exploitation balance.
    Rollout = construct workflow from decisions, evaluate on
    rollout_samples from benchmark, return mean fitness.

    Tree is persistent across generations — knowledge accumulates.
    """

    async def step(self, population, fitness_scores, trajectories) -> Population:
        # Run max_iterations of: select → expand → rollout → backpropagate
        # Extract top-K topologies from tree
        # Apply to non-elite workflows in population
        ...

    def get_statistics(self) -> dict:
        # tree size, visit counts, best path
        ...
```

**文件**: `src/optimizer/prompt_breeding.py`

```python
class PromptBreedingConfig(OptimizerConfig):
    breeding_model: str = "gpt-4o"       # LLM for prompt breeding
    num_failure_examples: int = 3        # failure trajectories shown to LLM


class PromptBreedingOptimizer(Optimizer):
    """LLM-driven semantic prompt crossover and mutation.

    Given two parent prompts with their fitness scores and failure
    trajectories, asks LLM to produce a child prompt that:
    1. Combines strengths of both parents
    2. Specifically addresses observed failure modes
    3. Preserves structural elements that correlate with high fitness

    Unlike GA string crossover, this operates at the semantic level —
    the LLM understands the meaning of prompt components and can
    intelligently recombine them.
    """

    async def step(self, population, fitness_scores, trajectories) -> Population:
        # Rank workflows by fitness
        # For each non-elite workflow:
        #   Select 2 parents (tournament)
        #   Gather failure trajectories for both parents
        #   Ask breeding_model to produce child prompt
        #   Validate child with 1-sample sanity check
        # Return updated population
        ...

    async def _breed(self, parent1_prompt: str, parent2_prompt: str,
                     parent1_failures: list[Trajectory],
                     parent2_failures: list[Trajectory]) -> str:
        """Ask LLM to cross two parent prompts."""
        ...

    def get_statistics(self) -> dict:
        ...
```

**文件**: `src/optimizer/hierarchical.py`

```python
class HierarchicalConfig(OptimizerConfig):
    """Configuration for the hierarchical hybrid optimizer."""
    micro_config: CMAESConfig = CMAESConfig()
    meso_config: PromptBreedingConfig = PromptBreedingConfig()
    macro_config: MCTSConfig = MCTSConfig()
    meso_interval: int = 2          # run meso every N generations
    macro_interval: int = 5         # run macro every N generations


class HierarchicalOptimizer(Optimizer):
    """Hierarchical hybrid optimizer — recommended default.

    Combines level-specific optimizers for maximum efficiency:
      - Micro (every gen):    CMA-ES for numeric hyperparameters
                              (temperature, max_steps, model selection)
      - Meso  (every 2 gens): PromptBreeding + TextGrad for
                              prompts, roles, actions
      - Macro (every 5 gens): MCTS for DAG topology search

    Each sub-optimizer uses the algorithm best suited for its search
    space, replacing the one-size-fits-all GA approach:

    | Level | Search space         | GA problem            | Replacement   | Advantage              |
    |-------|---------------------|-----------------------|---------------|------------------------|
    | Micro | Continuous/discrete | Random mutation       | CMA-ES        | ~10x sample efficiency |
    | Meso  | Natural language    | No semantic awareness | PromptBreeding| Semantic recombination |
    | Macro | Combinatorial DAG   | Blind random search   | MCTS          | Structured exploration |

    The schedule (meso_interval, macro_interval) controls how often
    expensive operations run. Micro runs every generation since CMA-ES
    has zero LLM cost (pure numerical computation).
    """

    def __init__(self, config: HierarchicalConfig, population: Population):
        super().__init__(config, population)
        self.micro = CMAESOptimizer(config.micro_config, population)
        self.meso = PromptBreedingOptimizer(config.meso_config, population)
        self.macro = MCTSOptimizer(config.macro_config, population)

    async def step(self, population, fitness_scores, trajectories) -> Population:
        gen = population.generation

        # Micro: always run (zero LLM cost, fast convergence on hyperparams)
        population = await self.micro.step(population, fitness_scores, trajectories)

        # Meso: prompt optimization (moderate LLM cost)
        if gen % self.config.meso_interval == 0:
            population = await self.meso.step(population, fitness_scores, trajectories)

        # Macro: topology search (highest cost, run infrequently)
        if gen % self.config.macro_interval == 0:
            population = await self.macro.step(population, fitness_scores, trajectories)

        return population

    def get_statistics(self) -> dict:
        return {
            "micro": self.micro.get_statistics(),
            "meso": self.meso.get_statistics(),
            "macro": self.macro.get_statistics(),
        }
```

#### 3.5 优化器注册表

**文件**: `src/optimizer/registry.py`

```python
class OptimizerRegistry:
    """Factory/registry for optimizers using ClassVar pattern (same as AsyncLLM).

    Usage:
        OptimizerRegistry.register("evolutionary", EvolutionaryOptimizer)
        OptimizerRegistry.register("llm", LLMOptimizer)
        optimizer = OptimizerRegistry.create("evolutionary", config, population)
    """
    _registry: ClassVar[dict[str, Type[Optimizer]]] = {}

    @classmethod
    def register(cls, name: str, optimizer_class: Type[Optimizer]) -> None:
        """Register an optimizer class under a name."""
        ...

    @classmethod
    def create(cls, name: str, config: OptimizerConfig, population: Population) -> Optimizer:
        """Create an optimizer instance by name."""
        ...

    @classmethod
    def list_optimizers(cls) -> list[str]:
        """List all registered optimizer names."""
        ...
```

默认注册：

| 注册名 | 类 |
|-------|------|
| `"evolutionary"` | `EvolutionaryOptimizer` |
| `"llm"` | `LLMOptimizer` |
| `"rl"` | `RLOptimizer` |
| `"self_refinement"` | `SelfRefinementOptimizer` |
| `"text_grad"` | `TextGradOptimizer` |
| `"cmaes"` | `CMAESOptimizer` |
| `"mcts"` | `MCTSOptimizer` |
| `"prompt_breeding"` | `PromptBreedingOptimizer` |
| **`"hierarchical"`** | **`HierarchicalOptimizer`** (推荐默认) |

---

### 阶段 4：工作流系统（Workflow System）

**目标**：实现可组合、可执行、可序列化、可演化的工作流 DAG，替代原有的交互模式。

**新目录**: `src/workflow/`（替代 `src/interaction/`）

Workflow is the core unit of SEMA: simultaneously the execution plan (DAG of agent nodes) and the evolutionary genome (what the optimizer mutates/selects). It is fully JSON-serializable via Pydantic.

#### 4.1 Schema (`src/workflow/schema.py`)

**文件**: `src/workflow/schema.py`

```python
class Role(BaseModel):
    """Defines an agent's persona and capabilities within a workflow node."""
    name: str                      # e.g., "researcher", "critic", "planner"
    description: str               # human-readable
    system_prompt: str             # persona definition
    available_tools: list[str] = []
    memory_strategy: str = "recent"  # "recent" | "relevant" | "hybrid"


class Action(BaseModel):
    """Defines what an agent does at a workflow node."""
    name: str                      # e.g., "search", "reason", "critique", "synthesize"
    instruction_prompt: str        # task-specific instruction
    output_schema: dict = {}       # expected output structure
    max_steps: int = 5


class WorkflowNode(BaseModel):
    """A node in the workflow DAG: an agent with a role performing an action."""
    node_id: str
    role: Role
    action: Action
    agent_config: dict = {}        # model, temperature, etc. → AgentConfig extras


class WorkflowEdge(BaseModel):
    """Directed edge: data/control flow between nodes."""
    source_id: str
    target_id: str
    edge_type: str = "data"        # "data" | "conditional" | "loop"
    condition: str | None = None   # for conditional edges
    data_mapping: dict = {}        # maps source output fields → target input fields
    max_iterations: int = 1        # for loop edges


class Workflow(BaseModel):
    """Executable multi-agent workflow and evolvable genome.

    The core unit of SEMA: simultaneously the execution plan (DAG)
    and the evolutionary genome (what the optimizer mutates/selects).
    Fully JSON-serializable via Pydantic.
    """
    workflow_id: str
    generation: int = 0
    parent_ids: list[str] = []
    goal: str                       # high-level objective
    environment: str                # environment/benchmark name
    nodes: list[WorkflowNode] = []
    edges: list[WorkflowEdge] = []
    entry_nodes: list[str] = []
    exit_nodes: list[str] = []
    fitness: float = 0.0
    fitness_history: list[float] = []
    metadata: dict = {}

    # Serialization
    def to_json(self, path: Path | None = None) -> str:
        """Serialize workflow to JSON string, optionally write to file."""
        ...

    @classmethod
    def from_json(cls, json_str: str | None = None, path: Path | None = None) -> Workflow:
        """Deserialize workflow from JSON string or file."""
        ...

    # Graph helpers
    def get_execution_order(self) -> list[list[str]]:
        """Topological sort into parallel execution layers."""
        ...

    def validate_graph(self) -> bool:
        """Check DAG validity (no cycles, entry/exit nodes exist)."""
        ...

    def get_node(self, node_id: str) -> WorkflowNode | None:
        """Retrieve a node by ID."""
        ...
```

**Blank workflow** (just a goal, no nodes):
```python
Workflow(workflow_id="...", goal="Answer the question accurately", environment="hotpotqa")
```
The optimizer bootstraps structure in its first `step()`. Or use a template as seed.

#### 4.2 Executor (`src/workflow/executor.py`)

**文件**: `src/workflow/executor.py`

```python
class WorkflowResult(BaseModel):
    """Result of executing a workflow on a single task."""
    workflow_id: str
    answer: str                     # final answer from exit nodes
    node_outputs: dict[str, Any]    # each node's output
    execution_trace: list[dict]     # detailed trace for feedback
    total_steps: int
    total_llm_calls: int


class WorkflowExecutor:
    """Executes a workflow DAG on a task.

    1. Topological sort → parallel layers
    2. For each layer: instantiate agents, run in parallel (asyncio.gather)
    3. Pass outputs between nodes via edge data_mapping
    4. Handle conditional edges (LLM-evaluated or expression) and loop edges
    5. Collect final output from exit nodes
    """
    async def execute(self, workflow: Workflow, task: dict) -> WorkflowResult:
        """Execute the workflow on a single task.

        For blank workflows (no nodes), returns a default/empty result.
        The optimizer is expected to add nodes before evaluation produces
        meaningful fitness.
        """
        ...

    def get_execution_trace(self) -> list[dict]:
        """Return the detailed execution trace."""
        ...
```

Internally, executor creates a lightweight `_NodeAgent(Agent)` for each node that wraps Role + Action into the Agent.run()/step() contract, reusing the existing AsyncLLM factory.

#### 4.3 Templates (`src/workflow/templates.py`)

**文件**: `src/workflow/templates.py`

```python
class WorkflowTemplate:
    """Predefined workflow topologies used as population seeds."""

    @staticmethod
    def blank(goal: str, environment: str) -> Workflow:
        """Empty: just goal, no nodes/edges. Optimizer builds from scratch."""
        ...

    @staticmethod
    def single_agent(goal: str, environment: str) -> Workflow:
        """One node: role='solver', action='reason'."""
        ...

    @staticmethod
    def chain(goal: str, environment: str) -> Workflow:
        """Sequential: planner → researcher → synthesizer."""
        ...

    @staticmethod
    def debate(goal: str, environment: str) -> Workflow:
        """Parallel proposers → critic → final judge."""
        ...

    @staticmethod
    def hierarchical(goal: str, environment: str) -> Workflow:
        """Coordinator → parallel workers → aggregator."""
        ...
```

#### 4.4 Environment (`src/workflow/environment.py`)

**文件**: `src/workflow/environment.py`

```python
class Environment(BaseModel):
    """Task environment for workflow execution and evolution."""
    name: str                       # identifier
    benchmark_name: str             # e.g., "hotpotqa", "gsm8k", "mbpp"
    dataset: str = "validate"       # dataset split
    description: str = ""           # natural language description
    constraints: dict = {}          # resource limits, etc.
    metadata: dict = {}

    def get_benchmark(self) -> Benchmark:
        """Instantiate the appropriate Benchmark from benchmark_name."""
        ...

    def has_changed(self, previous: Environment) -> bool:
        """Detect if environment differs from a previously saved one.

        Compares benchmark_name, dataset, and constraints.
        Used on resume to detect environment drift and trigger re-evaluation.
        """
        ...
```

When `SEMAOrchestrator.resume()` loads a checkpoint, it compares the current Environment to the saved one. If changed, it resets fitness scores and triggers re-evaluation before continuing optimization.

---

### 阶段 5：编排器（主循环）

**目标**：实现 SEMA 核心自演化循环，统合所有子系统。

**文件**: `src/orchestrator/config.py`, `src/orchestrator/experiment.py`, `src/orchestrator/orchestrator.py`

```python
# --- config.py ---
class SEMAConfig(BaseModel):
    experiment_name: str = "sema_experiment"
    storage_root: str = "~/.sema/experiments"
    environment: Environment                        # task environment (replaces benchmark_name + eval_dataset)
    optimizer_type: str = "evolutionary"            # registered optimizer name
    optimizer_config: OptimizerConfig                # optimizer-specific config
    seed_template: str = "single_agent"             # WorkflowTemplate name for initial population
    population_size: int = 10
    eval_samples_per_generation: int = 50           # samples per generation
    meta_reward: MetaRewardConfig                   # meta-reward config
    early_stop_generations: int = 10                # consecutive no-improvement threshold
    fitness_target: float | None = None             # target fitness


# --- experiment.py ---
class ExperimentTracker:
    """Experiment tracking: evolution history and metrics recording."""
    async def log_generation(self, generation: int, stats: dict, metrics: dict, best: Workflow) -> None
    async def save_checkpoint(self, population: Population, environment: Environment, generation: int) -> None
    async def load_checkpoint(self, generation: int = None) -> tuple[Population, Environment, int]
    def get_history(self) -> list[dict]
    def summary_report(self) -> str


# --- orchestrator.py ---
class SEMAOrchestrator:
    """SEMA main orchestrator."""
    async def run(self) -> dict
    async def resume(self, checkpoint_generation: int = None) -> dict
```

---

### 阶段 6：更多 Benchmark 支持

**目标**：扩展 benchmark 支持，增加 GSM8K 和 MBPP 评测基准。

#### 6.1 GSM8K

**文件**: `src/benchmarks/gsm8k.py`

`benchmarks.json` 已有 GSM8K 条目（train + test）。

```python
class GSM8K(Benchmark):
    """GSM8K benchmark for grade-school math reasoning.

    Evaluates numeric answer extraction and equivalence checking.
    GSM8K answers follow the format '#### <number>' in the dataset.
    """

    def __init__(self, data_folder: str = None, dataset_type: DatasetType = DatasetType.ALL):
        ...

    def load_data(self, force_reload: bool = False) -> None:
        # Same pattern as HotpotQA: read benchmarks.json, download if needed
        ...

    def load_dataset(self, dataset: dict | None, force_reload: bool = False) -> list[dict] | None:
        # Same pattern as HotpotQA
        ...

    async def evaluate(self, prediction: Any, label: Any) -> dict:
        """Extract numeric answer and check equivalence.

        Handles: integer/float comparison, comma-separated numbers,
        percentage notation, and basic unit normalization.
        """
        pred_num = self._extract_number(str(prediction))
        label_num = self._extract_number(str(label))
        em = 1.0 if pred_num is not None and pred_num == label_num else 0.0
        return {'em': em, 'f1': em, 'result': Benchmark.PASS if em == 1.0 else Benchmark.FAIL}

    def _extract_number(self, text: str) -> float | None:
        """Extract the final numeric answer from text.

        Looks for '#### <number>' pattern first (GSM8K format),
        then falls back to last number in text.
        """
        ...

    async def run(self, callback, dataset='test', num_samples=None, verbose=False) -> dict:
        # Same pattern as HotpotQA.run()
        ...
```

#### 6.2 MBPP

**文件**: `src/benchmarks/mbpp.py`

需要在 `benchmarks.json` 中增加 MBPP 条目。

```json
{
    "mbpp": {
        "test": {
            "url": "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl",
            "name": "mbpp_test.jsonl"
        }
    }
}
```

```python
class MBPP(Benchmark):
    """MBPP (Mostly Basic Python Problems) benchmark.

    Evaluates code generation by executing generated code against test cases.
    Each problem has a task description, reference solution, and test assertions.
    """

    def __init__(self, data_folder: str = None, dataset_type: DatasetType = DatasetType.ALL):
        ...

    def load_data(self, force_reload: bool = False) -> None:
        ...

    def load_dataset(self, dataset: dict | None, force_reload: bool = False) -> list[dict] | None:
        ...

    async def evaluate(self, prediction: Any, label: Any) -> dict:
        """Evaluate generated code by running test cases.

        Args:
            prediction: Generated Python code string.
            label: Dict with 'test_list' containing assertion strings.

        Returns:
            Dict with 'em' (all tests pass), 'f1' (fraction passed), 'result'.
        """
        passed, total = self._run_tests(str(prediction), label['test_list'])
        em = 1.0 if passed == total else 0.0
        f1 = passed / total if total > 0 else 0.0
        return {'em': em, 'f1': f1, 'result': Benchmark.PASS if em == 1.0 else Benchmark.FAIL}

    def _run_tests(self, code: str, test_list: list[str]) -> tuple[int, int]:
        """Execute code and run test assertions in a sandboxed environment.

        Returns (num_passed, num_total).
        """
        ...

    async def run(self, callback, dataset='test', num_samples=None, verbose=False) -> dict:
        # Same pattern as HotpotQA.run()
        ...
```

#### 6.3 Benchmark 导出更新

**文件**: `src/benchmarks/__init__.py` — 增加 GSM8K、MBPP 导出。

---

## 三、核心自演化循环

```
Input: config (SEMAConfig)

Initialization:
    environment = config.environment
    benchmark = environment.get_benchmark()
    seed_workflows = [WorkflowTemplate.create(config.seed_template, goal, environment.name)
                      for _ in range(config.population_size)]
    population = Population(size=config.population_size)
    population.initialize(seed_workflows)
    optimizer = OptimizerRegistry.create(config.optimizer_type, config.optimizer_config, population)
    meta_reward_computer = MetaRewardComputer(config.meta_reward)
    executor = WorkflowExecutor()
    experiment_tracker = ExperimentTracker(config.experiment_name)
    best_fitness = 0.0
    no_improve = 0

Main loop (generation = 0 .. max_generations):

    // === A: Evaluation ===
    fitness_scores = {}
    all_trajectories = []
    FOR EACH workflow IN population.workflows:

        trajectories = []
        FOR EACH (question, context, ground_truth) IN benchmark.sample(eval_samples):

            task = {"question": question, "context": context}
            result = await executor.execute(workflow, task)

            env_reward = benchmark.evaluate(result.answer, ground_truth)['f1']
            trajectory = Trajectory(
                workflow_id=workflow.workflow_id,
                question=question,
                context=context,
                steps=result.execution_trace,
                prediction=result.answer,
                ground_truth=ground_truth,
                env_reward=env_reward,
                node_outputs=result.node_outputs,
                total_llm_calls=result.total_llm_calls,
            )
            trajectory.meta_reward = meta_reward_computer.compute(trajectory)
            trajectories.append(trajectory)

        fitness_scores[workflow.workflow_id] = mean(t.meta_reward for t in trajectories)
        all_trajectories.extend(trajectories)

    // === B: Logging ===
    await experiment_tracker.log_generation(generation, population.summary(), fitness_scores)
    await experiment_tracker.save_checkpoint(population, environment, generation)

    // === C: Early stopping check ===
    IF best improved: reset no_improve
    ELSE: no_improve += 1
    IF no_improve >= threshold OR fitness >= target: BREAK

    // === D: Optimization (strategy-agnostic) ===
    population = await optimizer.step(population, fitness_scores, all_trajectories)
    // The optimizer internally applies its strategy:
    //   - EvolutionaryOptimizer: micro/meso/macro mutation + crossover + selection
    //   - LLMOptimizer: meta-LLM proposes improved workflows
    //   - RLOptimizer: policy gradient update
    //   - SelfRefinementOptimizer: self-critique and improvement
    //   - TextGradOptimizer: textual gradient backpropagation

Output: { best_workflow, final_fitness, generations, report, history }

Resume:
    // When resuming from checkpoint:
    population, saved_env, generation = experiment_tracker.load_checkpoint()
    IF environment.has_changed(saved_env):
        // Environment changed: reset fitness, re-evaluate before continuing
        reset_fitness(population)
    CONTINUE main loop from generation
```

## 四、数据流

```
Population (of Workflows)
    │
    ▼
WorkflowExecutor.execute(workflow, task)
    │
    ├─→ Topological sort → parallel execution layers
    │
    ├─→ Per-node: _NodeAgent instantiation (Role + Action → Agent)
    │       │
    │       ├─→ AsyncLLM (prompt → response)
    │       ├─→ ToolRegistry.execute()
    │       └─→ MemorySystem.recall()
    │
    ├─→ Inter-node: data flow via WorkflowEdge (data_mapping)
    │       │
    │       ├─→ Conditional edges (LLM-evaluated or expression)
    │       └─→ Loop edges (max_iterations)
    │
    ▼
WorkflowResult { answer, node_outputs, execution_trace }
    │
    ▼
Benchmark.evaluate(prediction, ground_truth)
    │           (HotpotQA: EM/F1, GSM8K: numeric match, MBPP: test execution)
    ▼
Trajectory { env_reward, execution_trace, node_outputs }
    │
    ▼
MetaRewardComputer.compute() → meta_reward
    │
    ├─→ FeedbackCollector.record()     → disk (JSONL)
    └─→ fitness_scores (per workflow)
            │
            ▼
    Optimizer.step(population, fitness_scores, trajectories)
        │
        │  (strategy depends on optimizer type)
        │  EvolutionaryOptimizer: Selection → Crossover → Mutation (micro/meso/macro)
        │  LLMOptimizer: Meta-LLM proposes improved workflows
        │  RLOptimizer: Policy gradient update
        │  SelfRefinementOptimizer: Self-critique → improvement
        │  TextGradOptimizer: Textual gradient → prompt update
        │
        ▼
    New Population (evolved Workflows) → next generation loop
```

## 五、存储结构

```
~/.sema/
├── logs/sema.log                         # existing
├── benchmarks/                              # existing
│   ├── hotpotqa_*.json
│   ├── gsm8k_*.jsonl                        # new
│   └── mbpp_*.jsonl                         # new
├── experiments/                             # new
│   └── {experiment_name}/
│       ├── config.json                      # SEMAConfig
│       ├── environment.json                 # saved Environment for resume
│       ├── history.jsonl                    # per-generation stats
│       ├── checkpoints/
│       │   ├── gen_000.json                 # population snapshot
│       │   └── ...
│       ├── workflows/
│       │   ├── gen_000/                     # per-generation workflow JSONs
│       │   │   ├── workflow_001.json
│       │   │   └── ...
│       │   └── best_workflow.json           # best workflow across all generations
│       ├── trajectories/
│       │   ├── gen_000.jsonl                # per-generation trajectories
│       │   └── ...
│       └── report.md                        # final report
├── memory/                                  # new
│   └── {agent_id}/
│       └── long_term.jsonl                  # long-term memory
└── tools/                                   # new (optional)
    └── custom_tools.json
```

## 六、新增/修改目录与文件

```
src/
├── agents/
│   ├── __init__.py                 # update: export new Agent methods
│   └── agent.py                   # MODIFY: extend AgentConfig/AgentState/Agent
├── memory/
│   ├── __init__.py                 # new
│   ├── base.py                     # new
│   ├── short_term.py               # new
│   ├── long_term.py                # new
│   └── memory_system.py            # new
├── tools/
│   ├── __init__.py                 # new
│   ├── base.py                     # new
│   ├── registry.py                 # new
│   └── builtin/
│       ├── __init__.py             # new
│       ├── search.py               # new
│       ├── calculator.py           # new
│       └── lookup.py               # new
├── feedback/
│   ├── __init__.py                 # new
│   ├── base.py                     # new
│   ├── meta_reward.py              # new
│   └── reward_shaping.py           # new
├── optimizer/
│   ├── __init__.py                 # new
│   ├── base.py                     # new (Optimizer ABC, OptimizerConfig)
│   ├── population.py               # new (Population — holds Workflows, no AgentGenome)
│   ├── evolutionary.py             # new (GA: micro/meso/macro mutation, crossover, selection)
│   ├── llm_optimizer.py            # new (OPRO-style)
│   ├── rl_optimizer.py             # new (RL-based)
│   ├── self_refinement.py          # new (iterative self-reflection)
│   ├── text_grad.py                # new (textual gradient)
│   ├── cmaes.py                    # new (CMA-ES for micro-level hyperparameters)
│   ├── mcts.py                     # new (MCTS for macro-level topology search)
│   ├── prompt_breeding.py          # new (LLM-driven semantic prompt crossover)
│   ├── hierarchical.py             # new (HierarchicalOptimizer — recommended default)
│   └── registry.py                 # new (OptimizerRegistry)
├── workflow/
│   ├── __init__.py                 # new
│   ├── schema.py                   # new (Role, Action, WorkflowNode, WorkflowEdge, Workflow)
│   ├── executor.py                 # new (WorkflowExecutor, WorkflowResult)
│   ├── templates.py                # new (WorkflowTemplate)
│   └── environment.py              # new (Environment)
├── orchestrator/
│   ├── __init__.py                 # new
│   ├── config.py                   # new
│   ├── experiment.py               # new
│   └── orchestrator.py             # new
├── benchmarks/
│   ├── __init__.py                 # update: export GSM8K, MBPP
│   ├── benchmarks.json             # update: add MBPP entry
│   ├── gsm8k.py                    # new
│   └── mbpp.py                     # new

tests/
├── memory/
│   ├── test_short_term.py
│   ├── test_long_term.py
│   └── test_memory_system.py
├── tools/
│   ├── test_registry.py
│   └── test_builtin.py
├── feedback/
│   ├── test_meta_reward.py
│   └── test_reward_shaping.py
├── optimizer/
│   ├── test_population.py
│   ├── test_evolutionary.py
│   ├── test_llm_optimizer.py
│   ├── test_cmaes.py
│   ├── test_mcts.py
│   ├── test_prompt_breeding.py
│   ├── test_hierarchical.py
│   └── test_registry.py
├── workflow/
│   ├── test_schema.py
│   ├── test_executor.py
│   ├── test_templates.py
│   └── test_environment.py
├── benchmarks/
│   ├── test_gsm8k.py
│   └── test_mbpp.py
└── orchestrator/
    ├── test_orchestrator.py
    └── test_experiment.py
```

## 七、阶段依赖关系

```
阶段 1 (基础层)
  Agent extension + Memory ─────────────┐
                                        │
阶段 2 (能力层)                         ├── no external deps, only depends on existing Agent layer
  Tool System ─────────── depends on Phase 1
  Feedback / Meta-Reward ─ depends on Phase 1 + existing Benchmark layer
                                        │
阶段 3 (优化层)                         │
  Optimizer Engine ────── depends on Phase 2 (Trajectory) + Phase 4 (Workflow)
                                        │
阶段 4 (工作流层)                       │
  Workflow System ─────── depends on Phase 1 (Agent) + existing infrastructure
                                        │
阶段 5 (编排层)                         │
  Orchestrator ────────── depends on all prior phases
                                        │
阶段 6 (评测扩展)                       │
  GSM8K + MBPP ────────── independent, can be implemented at any time
```

## 八、关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| Agent extension approach | Modify existing Agent directly (no subclass) | Keeps inheritance chain simple; evolution is opt-in via extra kwargs |
| AgentConfig extra policy | `extra='allow'` | Evolution fields passed as extra kwargs, no formal field declarations needed |
| Evolvable unit | Workflow (not AgentGenome) | Captures topology + roles + agent configs as one serializable genome |
| Workflow schema | Pydantic BaseModel | JSON serialization for free; validation; consistent with codebase |
| Role/Action separation | Two distinct models | Same role can perform different actions; both independently evolvable |
| Mutation levels | Micro/Meso/Macro | Fine-grained control: tune configs, rewrite prompts, or restructure topology |
| Blank workflow | Supported (no nodes) | Optimizer can bootstrap structure from goal alone |
| Optimization framework | Abstract `Optimizer(ABC)` with registry | Supports multiple strategies (EA, LLM, RL, self-refinement, TextGrad, CMA-ES, MCTS, PromptBreeding) via unified interface |
| **Default optimizer** | **`HierarchicalOptimizer`** (was: `EvolutionaryOptimizer`) | **Micro=CMA-ES + Meso=PromptBreeding + Macro=MCTS，各层用最优算法，整体比纯 GA 样本效率高、收敛快、成本低约 50%** |
| Prompt mutation | LLM-based PromptBreeding (not random GA crossover) | Semantic-level crossover with failure-aware recombination, far more effective than character-level |
| Memory retrieval | TF-IDF cosine similarity | Avoids vector DB dependency; extensible via `MemoryStore` subclasses |
| Environment tracking | Embedded in checkpoint | Enables environment-change detection on resume |
| Evaluation approach | Iterate dataset + reuse `Benchmark.evaluate()` | More flexible than `Benchmark.run()`; enables richer trajectory collection |
| Storage format | JSONL | Supports append writes and streaming reads; compatible with existing `load_json()` |
| Async consistency | All core methods async | Consistent with existing fully-async design |
| Benchmark coverage | HotpotQA + GSM8K + MBPP | Covers QA, math reasoning, and code generation |

## 九、潜在挑战与缓解

| 挑战 | 基础缓解措施 |
|------|---------|
| LLM API cost (population × samples × generations) | Use small models (gpt-4o-mini) for evaluation; optionally use larger models for mutation; reduce eval_samples |
| Slow evolution convergence | Elitism + reward shaping + multiple mutation operators in parallel; early stopping; multiple optimizer strategies |
| Unstable prompt mutation quality | Structured mutation instructions; constrain mutation magnitude; preserve template skeleton |
| Topology search space explosion | Constrain max nodes/edges; start from templates; macro-mutation rate low |
| Blank workflow bootstrapping | LLMOptimizer can analyze goal and propose initial structure; fallback to single_agent template |
| Environment drift detection | Compare benchmark_name + dataset + constraints; reset fitness on change |
| Memory bloat | Fixed capacity + importance-based eviction; periodic consolidation |
| Workflow execution latency | `asyncio.gather` for parallel execution layers; limit max nodes per workflow |
| Checkpoint consistency | Atomic writes (temp file then rename); generation-level checkpoints; environment.json for resume |
| MBPP code execution safety | Sandboxed execution with timeout and resource limits; restricted builtins |
| Optimizer selection | Default to evolutionary; `OptimizerRegistry` makes switching trivial |

### 9.1 详细可行方案

#### 9.1.1 LLM API 成本控制

- **分层评估 (Cascading Evaluation)**：不对所有个体都跑完整评估集。第一轮用 5-10 个样本快筛，淘汰底部 50%；只对存活个体跑完整 eval_samples。可将总调用量减少约 40-60%。实现上在 `SEMAOrchestrator` 评估阶段加两阶段筛选逻辑。
- **结果缓存 (Response Cache)**：对相同 prompt + question 组合做哈希缓存。演化早期很多个体 prompt 差异微小甚至相同，缓存命中率高。在 `AsyncLLM` 层加 LRU cache。
- **预算控制器 (Budget Controller)**：`SEMAConfig` 增加 `max_api_calls: int` 和 `max_cost_usd: float`，orchestrator 循环中实时累计，达阈值后强制停止并输出当前最优结果。

#### 9.1.2 演化收敛加速

- **Warm-start Population**：种群初始化时用 `WorkflowTemplate` 的 5 种模板各生成若干个体，再加随机扰动版本，提高初始多样性。
- **混合优化器 (Optimizer Ensemble)**：不同代之间交替使用不同优化器（奇数代 `EvolutionaryOptimizer` 全局搜索，偶数代 `SelfRefinementOptimizer` 局部改进），在 `SEMAOrchestrator` 层做调度。
- **Fitness Landscape Smoothing**：引入 curriculum——先在简单子集上评估，再逐步增加难度，让 fitness landscape 在早期更平滑。

#### 9.1.3 Prompt 变异质量稳定化

- **Diff-constrained Mutation**：变异指令要求 LLM 返回 JSON patch 格式的修改（非完整重写），限制 patch 数量（最多修改 2 句话）：
  ```
  Given the current prompt, suggest exactly 1-2 specific edits.
  Return as JSON: [{"action": "replace", "old": "...", "new": "..."}]
  Do NOT rewrite the entire prompt.
  ```
- **Mutation Validation**：变异后用 1 个标准问题做 sanity check，结果明显恶化则回滚。相当于低成本 "mutation gate"。
- **Temperature Annealing for Mutation**：变异用 LLM 的 temperature 随代数递减（早期 1.0 → 后期 0.3），早期鼓励探索，后期精细调优。

#### 9.1.4 拓扑搜索空间约束

- **Grammar-guided Topology Mutation**：定义合法拓扑操作原语（类似 grammar）：
  - `insert_parallel_branch(a, b)` — 在 a→b 之间插入并行分支
  - `insert_sequential_step(a, b)` — 在 a→b 之间串联新节点
  - `merge_nodes(a, b)` — 合并相邻节点
  `TopologyMutator` 从原语中采样，保证 DAG 合法性并限制搜索空间。
- **Topology Complexity Penalty**：MetaReward 加入拓扑复杂度惩罚项（如节点数的对数），形成对简洁拓扑的 inductive bias。

#### 9.1.5 Blank Workflow 冷启动

- **Mandatory Minimum Structure**：`Population.initialize()` 中若 seed workflow 为 blank，自动调用 `LLMOptimizer` 的 bootstrap 方法生成初始结构后再加入种群。确保不存在空白个体进入评估。
- **Goal-conditioned Template Selection**：根据 goal 和 environment 语义自动选择最合适的模板（goal 含 "debate" → debate 模板，含 "step by step" → chain 模板）。

#### 9.1.6 环境漂移检测增强

- **Data Fingerprint**：除比较配置字段外，对实际数据前 100 条做 hash 指纹，检测配置名未变但数据内容已变的情况。
- **Graduated Reset**：环境变化时将旧 fitness 乘以衰减因子（如 0.5），让好的个体仍有优势但须通过新环境评估"证明"自己，比完全重置收敛更快。

#### 9.1.7 Memory 膨胀控制

- **Tiered Eviction Policy**：三级淘汰——
  1. 短期记忆：FIFO 滑动窗口
  2. 长期记忆：达容量上限时按 `importance × recency_decay` 排序，淘汰底部 20%
  3. Archive 层：被淘汰记忆用 LLM 做 summarization 压缩为摘要记忆，信息不完全丢失
- **Per-generation Memory Reset**：每个 generation 开始时调用 `memory.reset_short_term()` 并可选做长期记忆 consolidation，防止跨代记忆污染。

#### 9.1.8 Workflow 执行延迟优化

- **Workflow-level Parallelism**：改为多个 workflow 并行评估，用 `asyncio.Semaphore` 控制并发度（同时评估 3-5 个 workflow），不超过 API rate limit 下大幅缩短每代 wall-clock time：
  ```python
  sem = asyncio.Semaphore(5)
  async def eval_with_limit(wf):
      async with sem:
          return await evaluate_workflow(wf)
  results = await asyncio.gather(*[eval_with_limit(wf) for wf in population.workflows])
  ```
- **Execution Timeout per Node**：每个节点设超时（如 30 秒），超时返回空结果不阻塞整个 workflow。用 `asyncio.wait_for()` 包装。

#### 9.1.9 Checkpoint 一致性保障

- **Write-Ahead Log (WAL)**：写 checkpoint 前先写 WAL 记录（含 generation 号和状态摘要）。resume 时检查 WAL——若最后 WAL 的 generation 比 checkpoint 新，回退到上一个完整 checkpoint：
  ```
  experiments/{name}/wal.jsonl:
  {"generation": 5, "status": "checkpoint_start", "timestamp": ...}
  {"generation": 5, "status": "checkpoint_complete", "timestamp": ...}
  ```
- **Multi-generation Retention**：保留最近 N 个（如 3 个）generation 的 checkpoint，旧的自动清理。

#### 9.1.10 MBPP 代码执行安全

- **subprocess + RestrictedPython 双重隔离**：
  1. `subprocess.run()` 子进程执行，`timeout=10` 秒
  2. 子进程内用 `RestrictedPython` 限制 builtins（禁 `open`, `exec`, `eval`, `__import__` 等）
  3. `resource` 模块（Linux）或 job object（Windows）限制内存（256MB）
- **Docker 容器沙箱**（可选高安全模式）：每次评估在临时 Docker 容器中执行，隔离最彻底。作为 `SEMAConfig` 的可选配置项。

#### 9.1.11 优化器自动选择

- **Auto-select with Short Probe**：正式演化前各优化器先跑 2-3 代 probe 阶段（极小 eval_samples，如 5 个），比较 fitness 提升速率，自动选最佳优化器。
- **Optimizer Switching Policy**：演化中监控 fitness plateau——连续 N 代无改善则自动切换到另一种优化器。利用注册表统一接口，切换成本为零。

### 9.2 优先级排序

以上方案按投入产出比排序，**最高优先级**的三条：

1. **分层评估 (Cascading Evaluation)** — 直接砍掉 40-60% 的 API 成本
2. **Workflow-level 并行评估** — 几行代码将每代 wall-clock time 缩短数倍
3. **Diff-constrained Mutation** — prompt 变异质量是演化能否有效运作的关键

### 9.3 GA 替代方案分析与分层混合优化器

#### 9.3.1 遗传算法（GA）在 SEMA 场景下的瓶颈

1. **样本效率低**：GA 是 black-box 优化，不利用任何梯度信息，每一代需要大量 fitness 评估（= LLM 调用）才能推动种群前进
2. **文本空间不适合传统变异/交叉**：prompt 是自然语言，随机扰动在离散语义空间中效果差，不像连续参数空间有平滑的 fitness landscape
3. **收敛慢**：搜索空间是 prompt 文本 + DAG 拓扑的组合，GA 的随机搜索在高维离散空间中很慢

#### 9.3.2 候选替代算法

**方案 A：Bayesian Optimization（贝叶斯优化）**

- **核心思想**：用代理模型（surrogate model）拟合 fitness landscape，指导下一步搜索
- **适用层级**：Micro-level 参数优化（temperature, max_steps, model 选择）
- **优势**：样本效率比 GA 高一个数量级（通常 20-50 次评估即可收敛）；天然适合 expensive black-box optimization
- **限制**：不适合高维文本空间，但可与 LLM-based prompt 优化组合
- **依赖**：scikit-optimize 或 optuna

**方案 B：CMA-ES（协方差矩阵自适应进化策略）** ← 推荐用于 Micro 层

- **核心思想**：维护多元正态分布，每代从中采样候选解，根据 fitness 更新分布均值和协方差矩阵
- **适用层级**：Micro-level 数值参数优化，中等维度（10-100 维）下最优
- **优势**：自适应步长（无需手动调 mutation rate）；连续空间收敛远快于 GA；种群规模小（4+3×ln(dim)）
- **限制**：不直接适用于文本空间
- **依赖**：cmaes 或 pycma

**方案 C：MCTS（蒙特卡洛树搜索）** ← 推荐用于 Macro 层

- **核心思想**：把 workflow 拓扑构造建模为决策树（添加什么 Role？用什么 Action？连什么边？），用 UCB1 策略平衡探索与利用
- **适用层级**：Macro-level 拓扑搜索
- **优势**：有理论保证的探索-利用平衡；信息复用效率高（每次 rollout 更新整棵搜索树）；天然适合组合搜索空间
- **限制**：rollout 需要实际评估 workflow，有一定成本

**方案 D：Prompt Breeding（语义级 prompt 交叉）** ← 推荐用于 Meso 层

- **核心思想**：给 LLM 两个 parent prompt + 各自的 fitness + 失败案例，让 LLM 语义级"交叉"出子代 prompt
- **适用层级**：Meso-level prompt/role/action 优化
- **优势**：语义理解级别的交叉，远优于 GA 的字符串级操作
- **限制**：每次交叉需要一次 meta-LLM 调用

#### 9.3.3 推荐方案：分层混合优化器（Hierarchical Hybrid Optimizer）

不用一个算法替换 GA，而是在不同层级用最适合的算法：

| 优化层级 | 当前方案 (GA) | 推荐替代 | 理由 |
|---------|-------------|---------|------|
| **Micro** (temperature, max_steps, model) | GA 随机变异 | **CMA-ES** | 连续/离散超参数，样本效率高一个数量级 |
| **Meso** (prompt, role, action) | GA + LLM rewrite | **PromptBreeding + TextGrad** | 文本空间需语义级操作，非随机扰动 |
| **Macro** (DAG topology) | GA 随机拓扑变异 | **MCTS** | 组合搜索空间，需结构化探索 |

详细实现见 **阶段 3 → 3.4 新增优化器实现** (`src/optimizer/hierarchical.py`)。

**相比纯 GA 的预期改善**：
- 总代数减少约 40%（~25 代 vs ~40 代），因各层用最高效算法
- API 成本降低约 50%（CMA-ES 零 LLM 成本 + 更少代数）
- Prompt 优化质量更高（语义交叉 vs 随机变异）
- 拓扑搜索更有方向性（MCTS 结构化探索 vs 随机增删）

---

## 十、HotpotQA 成本预估（HierarchicalOptimizer 方案）

### 10.1 基础假设

| 参数 | 值 | 说明 |
|------|------|------|
| `population_size` | 10 | 种群大小 |
| `eval_samples_per_generation` | 50 | 每代评估样本数 |
| 有效代数 | ~25 | Hierarchical 更快收敛（纯 GA 约需 40 代） |
| 平均 workflow 节点数 | 3 | 从 1 逐步增长到 ~4，取平均 |
| 分层评估 (Cascading) | 开启 | Stage 1: 10 样本筛 50%，Stage 2: 50 样本评估存活个体 |
| `meso_interval` | 2 | 每 2 代做一次 prompt 优化 |
| `macro_interval` | 5 | 每 5 代做一次拓扑搜索 |
| 精英率 | 20% (2/10) | 每代 8 个非精英个体需要优化 |

### 10.2 LLM 调用量估算

#### A. 评估阶段（Worker LLM）— 每代

| 阶段 | 计算 | 调用数/代 |
|------|------|----------|
| Stage 1 快筛 | 10 workflows × 10 samples × 3 nodes | 300 |
| Stage 2 精评 | 5 workflows × 50 samples × 3 nodes | 750 |
| **小计** | | **1,050** |

25 代总计：**26,250 次**

#### B. 优化阶段（Meta-LLM）

| 操作 | 触发频率 | 每次调用数 | 总轮数 | 总调用数 |
|------|---------|-----------|--------|---------|
| Meso: PromptBreeding | 每 2 代 | 8 workflows × 1 call | 13 轮 | 104 |
| Meso: TextGrad gradient | 每 2 代 | 8 workflows × 1 call | 13 轮 | 104 |
| Macro: MCTS 节点提议 | 每 5 代 | 10 iterations × 1 call | 5 轮 | 50 |
| **Meta-LLM 小计** | | | | **258** |

#### C. 优化阶段附带的 Worker LLM 调用

| 操作 | 计算 | 总调用数 |
|------|------|---------|
| Meso mutation validation | 13 轮 × 8 workflows × 1 sample × 3 nodes | 312 |
| Macro MCTS rollout | 5 轮 × 10 iterations × 3 samples × 3 nodes | 450 |
| **附带 Worker 小计** | | **762** |

#### D. 总调用量汇总

| LLM 角色 | 总调用数 |
|----------|---------|
| **Worker LLM**（评估 + 优化附带） | 26,250 + 762 = **27,012** |
| **Meta-LLM**（优化决策） | **258** |
| **合计** | **27,270** |

### 10.3 Token 用量估算

#### Worker LLM（HotpotQA 场景）

| 方向 | 组成 | 平均 tokens/call |
|------|------|-----------------|
| Input | system_prompt(200) + context(1000) + question(50) + node_instruction(150) + prev_node_output(200) | **~1,600** |
| Output | reasoning(200) + answer(100) | **~300** |

- Total input: 27,012 × 1,600 = **43.2M tokens**
- Total output: 27,012 × 300 = **8.1M tokens**

#### Meta-LLM（优化决策）

| 方向 | 组成 | 平均 tokens/call |
|------|------|-----------------|
| Input | optimization_instruction(500) + parent_prompts/trajectories(2000) + fitness_summary(200) | **~2,700** |
| Output | new_prompt / gradient / topology_proposal | **~400** |

- Total input: 258 × 2,700 = **0.70M tokens**
- Total output: 258 × 400 = **0.10M tokens**

#### Token 总量

| LLM 角色 | Input | Output | Total |
|----------|-------|--------|-------|
| Worker LLM | 43.2M | 8.1M | 51.3M |
| Meta-LLM | 0.70M | 0.10M | 0.80M |
| **合计** | **43.9M** | **8.2M** | **52.1M** |

### 10.4 各模型配置成本对比

> 定价基于 2025 年公开 API 价格，实际可能有变动。

#### 配置 A：gpt-4o-mini (Worker) + gpt-4o (Meta) — 推荐性价比方案

| LLM | 角色 | Input 单价 | Output 单价 | Input 费用 | Output 费用 | 小计 |
|-----|------|-----------|------------|-----------|------------|------|
| gpt-4o-mini | Worker | $0.15/M | $0.60/M | $6.48 | $4.86 | **$11.34** |
| gpt-4o | Meta | $2.50/M | $10.00/M | $1.75 | $1.00 | **$2.75** |
| | | | | | **总计** | **$14.09** |

#### 配置 B：gpt-4o (Worker) + gpt-4o (Meta) — 高质量方案

| LLM | 角色 | Input 单价 | Output 单价 | Input 费用 | Output 费用 | 小计 |
|-----|------|-----------|------------|-----------|------------|------|
| gpt-4o | Worker | $2.50/M | $10.00/M | $108.00 | $81.00 | **$189.00** |
| gpt-4o | Meta | $2.50/M | $10.00/M | $1.75 | $1.00 | **$2.75** |
| | | | | | **总计** | **$191.75** |

#### 配置 C：Claude 3.5 Haiku (Worker) + Claude 3.5 Sonnet (Meta)

| LLM | 角色 | Input 单价 | Output 单价 | Input 费用 | Output 费用 | 小计 |
|-----|------|-----------|------------|-----------|------------|------|
| Claude 3.5 Haiku | Worker | $0.80/M | $4.00/M | $34.56 | $32.40 | **$66.96** |
| Claude 3.5 Sonnet | Meta | $3.00/M | $15.00/M | $2.10 | $1.50 | **$3.60** |
| | | | | | **总计** | **$70.56** |

#### 配置 D：DeepSeek-V3 (Worker) + DeepSeek-V3 (Meta) — 最低成本方案

| LLM | 角色 | Input 单价 | Output 单价 | Input 费用 | Output 费用 | 小计 |
|-----|------|-----------|------------|-----------|------------|------|
| DeepSeek-V3 | Worker | $0.27/M | $1.10/M | $11.66 | $8.91 | **$20.57** |
| DeepSeek-V3 | Meta | $0.27/M | $1.10/M | $0.19 | $0.11 | **$0.30** |
| | | | | | **总计** | **$20.87** |

#### 配置 E：Gemini 2.0 Flash (Worker) + gpt-4o (Meta) — 极致性价比方案

| LLM | 角色 | Input 单价 | Output 单价 | Input 费用 | Output 费用 | 小计 |
|-----|------|-----------|------------|-----------|------------|------|
| Gemini 2.0 Flash | Worker | $0.10/M | $0.40/M | $4.32 | $3.24 | **$7.56** |
| gpt-4o | Meta | $2.50/M | $10.00/M | $1.75 | $1.00 | **$2.75** |
| | | | | | **总计** | **$10.31** |

### 10.5 成本汇总对比

| 配置 | Worker 模型 | Meta 模型 | 总成本 | 相对成本 |
|------|-----------|----------|--------|---------|
| **E (极致性价比)** | Gemini 2.0 Flash | gpt-4o | **$10** | 1.0x |
| **A (推荐)** | gpt-4o-mini | gpt-4o | **$14** | 1.4x |
| **D (最低成本)** | DeepSeek-V3 | DeepSeek-V3 | **$21** | 2.0x |
| **C (Anthropic)** | Claude 3.5 Haiku | Claude 3.5 Sonnet | **$71** | 6.9x |
| **B (高质量)** | gpt-4o | gpt-4o | **$192** | 18.7x |

### 10.6 与纯 GA 方案的成本对比（配置 A 为例）

| 方案 | 有效代数 | Worker 调用 | Meta 调用 | 总成本 | 节省 |
|------|---------|-----------|----------|--------|------|
| 纯 GA（无 Cascading） | ~40 | 60,000 | ~320 | **$28** | — |
| **HierarchicalOptimizer + Cascading** | ~25 | 27,012 | 258 | **$14** | **~50%** |

### 10.7 注意事项

- 以上为中位估计，实际成本可能因 workflow 节点数增长（4-5 个节点）、HotpotQA context 长度变化、缓存命中率等因素上下浮动 ±30%
- 开启 **Response Cache** 后，早期演化中相同/相似 prompt 的缓存命中可进一步减少 10-20% Worker 调用
- 使用 **Budget Controller** 可设置硬上限，避免意外超支
- 配置 D (DeepSeek) 虽然单价低，但 meta-LLM 也用 DeepSeek，prompt breeding 质量可能不如 gpt-4o；混搭（Worker=DeepSeek, Meta=gpt-4o）可能是更优折中，预估总成本约 $24