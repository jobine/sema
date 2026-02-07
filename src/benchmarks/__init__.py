# Imports are done explicitly to avoid circular import issues
# when running modules directly with `python -m src.benchmarks.xxx`
#
# Usage:
#   from src.benchmarks.hotpotqa import HotpotQA
#   from src.benchmarks.benchmark import Benchmark, DatasetType

from .benchmark import Benchmark, DatasetType
from .hotpotqa import HotpotQA

__all__ = ['Benchmark', 'DatasetType', 'HotpotQA']