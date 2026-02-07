'''Async LLM helpers backed by JSON configuration.'''

from pathlib import Path
from typing import Any, ClassVar, Dict, Type
from .base_model import LLMConfig, AsyncBaseLLM
from .claude_model import AsyncAnthropicLLM
from .openai_model import AsyncOpenAILLM
from .ollama_model import AsyncOllamaLLM
from .gemini_model import AsyncGeminiLLM


class AsyncLLM:
	'''Factory class for creating async LLM instances based on provider.'''

	_provider_registry: ClassVar[Dict[str, Type[AsyncBaseLLM]]] = {
		'openai': AsyncOpenAILLM,
		'azure': AsyncOpenAILLM,
		'azure_openai': AsyncOpenAILLM,
		'ollama': AsyncOllamaLLM,
		'gemini': AsyncGeminiLLM,
		'google': AsyncGeminiLLM,
		'claude': AsyncAnthropicLLM,
		'anthropic': AsyncAnthropicLLM,
	}

	def __new__(
		cls,
		model: str,
		*,
		config_path: str | Path | None = None,
		**kwargs: Any,
	) -> AsyncBaseLLM:
		'''Create an async LLM instance based on the provider in config.

		Args:
			model: Model key defined in the JSON file.
			config_path: Optional override path to the JSON config file.
			**kwargs: Additional arguments passed to the LLM implementation.

		Returns:
			AsyncBaseLLM instance (AsyncOpenAILLM or AsyncGeminiLLM).
		'''
		config = LLMConfig.load(model, path=config_path)
		provider = config.provider.lower()

		llm_class = cls._provider_registry.get(provider)
		if llm_class is None:
			# Default to OpenAI-compatible API for unknown providers
			llm_class = AsyncOpenAILLM

		return llm_class(config, **kwargs)

	@classmethod
	def register_provider(cls, provider: str, llm_class: Type[AsyncBaseLLM]) -> None:
		'''Register a custom LLM implementation for a provider.

		Args:
			provider: Provider name (case-insensitive).
			llm_class: LLM class that extends AsyncBaseLLM.
		'''
		cls._provider_registry[provider.lower()] = llm_class


if __name__ == '__main__':
	# Example usage
	import asyncio

	async def main():
		# # OpenAI example
		# openai_llm = AsyncLLM('gpt-4o-mini')
		# result = await openai_llm('Hello, OpenAI!')
		# print(result)

		# # Gemini example (uncomment if configured)
		# gemini_llm = AsyncLLM('gemini-3-pro-preview')
		# result = await gemini_llm('Hello, Gemini!')
		# print(result)

		# Claude example (uncomment if configured)
		claude_llm = AsyncLLM('claude-opus-4-5')
		result = await claude_llm('Hello, Claude!')
		print(result)

	asyncio.run(main())
