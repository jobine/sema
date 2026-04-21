# SEMA: Self-Evolving Multi-Agent Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

**SEMA** is a cutting-edge framework for lifelong, self-evolving multi-agent systems. Unlike static agent architectures, SEMA introduces a continuous evolution loop where a population of agents—represented as evolvable **Workflows**—continually refines their reasoning processes, prompt strategies, memory utilization, and tool-use capabilities based on environmental feedback and meta-rewards.

## 🌟 Key Features

- **Lifelong Evolution**: Agents don't just perform tasks; they learn from their successes and failures to improve their future performance.
- **Multi-Level Mutation**: Evolves at three distinct levels:
  - **Micro**: Tweaking agent hyperparameters (temperature, max steps).
  - **Meso**: Refining semantic content (system prompts, instructions, tool strategies) using LLMs.
  - **Macro**: Reshaping the interaction topology (adding/removing nodes and edges in the workflow DAG).
- **Dual-Layer Memory**: Combines task-specific `Short-Term Memory` with persistent, cross-generation `Long-Term Memory`.
- **Advanced Optimization Engine**: A plug-and-play architecture supporting multiple optimization paradigms including Genetic Algorithms (EA), LLM-as-Optimizer (OPRO), Reinforcement Learning (RL), Self-Refinement, and Textual Gradients (TextGrad).
- **Unified Workflow Abstraction**: Workflows are defined as JSON-serializable Directed Acyclic Graphs (DAGs), making them easy to inspect, mutate, and evolve.

## 🏗 Architecture Overview

SEMA operates through a sophisticated orchestration loop that integrates optimization, workflow execution, feedback collection, and memory management.

```text
┌─────────────────────────────────────────────────────────────────────┐
│                       SEMA Orchestrator                              │
│         (self-evolving main loop, lifecycle, experiment tracking)    │
├───────────┬───────────┬────────────┬────────────┬──────────────────┤
│ Optimization│  Workflow  │ Feedback & │   Memory   │     Tool         │
│  Engine    │  Engine    │ Meta-Reward│   System   │     System       │
│            │            │            │            │                  │
│ - Optimizer│ - Schema   │ - Env.     │ - Short    │ - Registry       │
│   Registry │ - Executor │   feedback │   term     │ - Execution      │
│ - EA / LLM │ - Template │ - Meta-    │ - Long     │ - Strategy       │
│ / RL / SR  │ - Environ. │   reward   │   term     │   evolution      │
│ / TextGrad │            │ - Shaping  │ - Retrieval│                  │
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

## 🛠 Core Modules

### 🧠 Agents & Workflows
- **`src/agents`**: Defines the intelligent entities. Agents are equipped with evolvable configurations and state tracking.
- **`src/workflow`**: The "Genome" of SEMA. Workflows are executable DAGs where each node represents a specific Role or Action. Evolution directly manipulates these graph structures.

### 🚀 Optimization Engine (`src/optimizer`)
A modular suite of optimization strategies:
- **Evolutionary (EA)**: Classic Genetic Algorithms with micro/meso/macro mutations.
- **LLM-based**: Using LLMs to hypothesize and generate improved workflows (OPRO-style).
- **Reinforcement Learning (RL)**: Policy-based optimization guided by meta-rewards.
- **Self-Refinement**: Agents iteratively critique and improve their own execution traces.
- **TextGrad**: Treating semantic feedback as a "gradient" to update prompts.
- **CMA-ES & MCTS**: Sophisticated mathematical and search-based optimization methods.

### 💾 Memory System (`src/memory`)
- **Short-Term**: A sliding-window memory for task-local context.
- **Long-Term**: A persistent repository for cross-task knowledge and historical successes.

### 🔄 Feedback & Meta-Reward (`src/feedback`)
Automatically synthesizes multi-dimensional feedback (Accuracy, Efficiency, Tool usage, Memory utility) into a scalar **Meta-Reward** to guide the optimizer.

### 🛠 Tool System (`src/tools`)
A robust registry allowing agents to interact with the world (e.g., Calculators, Web Search, Database Lookups).

## 🚀 Getting Started

### Prerequisites
- Python 3.12 or higher
- API Keys for your preferred LLMs (OpenAI, Anthropic, Google, etc.)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/sema.git
cd sema

# Install dependencies using poetry
poetry install
```

### Basic Usage

```python
from src.orchestrator import SEMAOrchestrator
from src.optimizer import EvolutionaryOptimizer

# Initialize the orchestrator with an optimizer
orchestrator = SEMAOrchestrator(
    optimizer=EvolutionaryOptimizer(population_size=10),
    benchmark="hotpotqa"
)

# Start the self-evolution loop
orchestrator.run(generations=5)
```

## 📂 Project Structure

```text
sema/
├── examples/           # Practical usage examples
├── src/
│   ├── agents/         # Agent definitions and evolution support
│   ├── benchmarks/     # Evaluation environments
│   ├── config/         # Configuration management
│   ├── feedback/       # Feedback collection & Meta-reward computation
│   ├── memory/         # Short-term and Long-term memory systems
│   ├── models/         # LLM interface abstractions
│   ├── optimizer/      # Diverse optimization strategies (EA, RL, LLM, etc.)
│   ├── orchestrator/   # Main evolution loop and lifecycle management
│   ├── tools/          # Tool registry and builtin tools
│   ├── utils/          # Helper functions and logging
│   └── workflow/       # Workflow DAG definitions and schema
├── pyproject.toml      # Project metadata and dependencies
└── README.md
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
