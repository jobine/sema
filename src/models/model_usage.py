'''Thread-safe singleton for tracking LLM token usage and cost.'''

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from ..utils import get_logger

logger = get_logger(__name__)

_DEFAULT_PRICING_PATH = Path.home() / '.sema' / 'pricing' / 'model_prices.json'


class ModelUsage:
	'''Global singleton that accumulates prompt/completion tokens and cost.

	Thread-safe: all mutations are guarded by a threading.Lock.
	'''

	_instance: ModelUsage | None = None
	_init_lock: threading.Lock = threading.Lock()

	def __init__(self, pricing_path: Path | None = None) -> None:
		self._lock = threading.Lock()
		self._pricing_path = pricing_path or _DEFAULT_PRICING_PATH
		self._prices: dict[str, dict[str, Any]] = {}
		self._total_prompt_tokens: int = 0
		self._total_completion_tokens: int = 0
		self._total_cost: float = 0.0
		self._load_prices()

	def _load_prices(self) -> None:
		'''Load pricing data from local JSON cache.'''
		if not self._pricing_path.exists():
			logger.warning(f'Pricing file not found: {self._pricing_path}. All costs will be $0.')
			return
		try:
			with open(self._pricing_path, 'r', encoding='utf-8') as f:
				self._prices = json.load(f)
			logger.info(f'Loaded pricing for {len(self._prices)} models from {self._pricing_path}')
		except (json.JSONDecodeError, OSError) as exc:
			logger.warning(f'Failed to load pricing: {exc}. All costs will be $0.')

	@classmethod
	def get_instance(cls, pricing_path: Path | None = None) -> ModelUsage:
		'''Return the global singleton, creating it on first call.'''
		if cls._instance is None:
			with cls._init_lock:
				if cls._instance is None:
					cls._instance = cls(pricing_path=pricing_path)
		return cls._instance

	def _compute_cost(self, model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
		'''Compute cost for a single LLM call based on pricing data.'''
		price_info = self._prices.get(model_id)
		if price_info is None:
			return 0.0
		input_cost = prompt_tokens * price_info.get('input_cost_per_token', 0.0)
		output_cost = completion_tokens * price_info.get('output_cost_per_token', 0.0)
		return input_cost + output_cost

	def record(self, model_id: str, prompt_tokens: int, completion_tokens: int) -> None:
		'''Record one LLM call: compute cost, log, accumulate.'''
		cost = self._compute_cost(model_id, prompt_tokens, completion_tokens)
		with self._lock:
			self._total_prompt_tokens += prompt_tokens
			self._total_completion_tokens += completion_tokens
			self._total_cost += cost
		logger.info(
			f'LLM usage: model={model_id}, prompt_tokens={prompt_tokens}, '
			f'completion_tokens={completion_tokens}, cost=${cost:.6f}'
		)

	@property
	def total_prompt_tokens(self) -> int:
		return self._total_prompt_tokens

	@property
	def total_completion_tokens(self) -> int:
		return self._total_completion_tokens

	@property
	def total_cost(self) -> float:
		return self._total_cost

	def summary(self) -> str:
		'''Formatted summary string for console output.'''
		return (
			f'=== Model Usage Summary ===\n'
			f'Total prompt tokens:     {self._total_prompt_tokens:,}\n'
			f'Total completion tokens:  {self._total_completion_tokens:,}\n'
			f'Total cost:              ${self._total_cost:.4f}'
		)

	def report_section(self) -> str:
		'''Markdown section for report.md.'''
		return (
			f'## Model Usage\n\n'
			f'| Metric | Value |\n'
			f'|--------|-------|\n'
			f'| Total prompt tokens | {self._total_prompt_tokens:,} |\n'
			f'| Total completion tokens | {self._total_completion_tokens:,} |\n'
			f'| Total cost | ${self._total_cost:.4f} |\n'
		)
