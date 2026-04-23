import os
from collections.abc import AsyncIterator
from typing import Any, Dict
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from ollama import AsyncClient as AsyncOllama
from .base import LLMConfig, AsyncBaseLLM
from .model_usage import ModelUsage


class AsyncOllamaLLM(AsyncBaseLLM):
	'''Async wrapper for Ollama local models using official SDK with proxy bypass.'''

	def __init__(
		self,
		config: LLMConfig,
		*,
		client: AsyncOllama | None = None,
	) -> None:
		super().__init__(config)

		if client:
			self._client = client
		else:
			# Parse host from base_url if provided, otherwise use default localhost:11434
			host = 'http://localhost:11434'
			if self.config.base_url:
				# Remove /v1 suffix if present (OpenAI compatibility endpoint)
				host = self.config.base_url.rstrip('/').removesuffix('/v1')

			# Set NO_PROXY to bypass system proxy for localhost
			no_proxy = os.environ.get('NO_PROXY', '')
			if 'localhost' not in no_proxy and '127.0.0.1' not in no_proxy:
				entries = [e for e in no_proxy.split(',') if e]
				entries.extend(['localhost', '127.0.0.1'])
				os.environ['NO_PROXY'] = ','.join(entries)

			self._client = AsyncOllama(host=host)

	@retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10), retry=retry_if_exception_type(Exception))
	async def __call__(self, prompt: str, **kwargs: Any) -> str | AsyncIterator[str]:
		'''Call Ollama LLM. Returns str or AsyncIterator[str] for streaming.'''
		# Build options dict for Ollama
		options: Dict[str, Any] = {}
		if self.config.temperature is not None:
			options['temperature'] = self.config.temperature

		# Allow override from kwargs
		options_override: Dict[str, Any] = kwargs.pop('options', {})
		options.update(options_override)

		# Build messages format
		messages = [{'role': 'user', 'content': prompt}]
		messages_override = kwargs.pop('messages', None)
		if messages_override:
			messages = messages_override

		stream = kwargs.pop('stream', False)

		if stream:
			async def _gen() -> AsyncIterator[str]:
				response = await self._client.chat(
					model=self.config.id,
					messages=messages,
					options=options if options else None,
					stream=True,
					**kwargs,
				)
				async for chunk in response:
					if isinstance(chunk, dict):
						done = chunk.get('done', False)
						content = chunk.get('message', {}).get('content', '')
						if done:
							prompt_tokens = chunk.get('prompt_eval_count', 0) or 0
							completion_tokens = chunk.get('eval_count', 0) or 0
							ModelUsage.get_instance().record(
								self.config.id, prompt_tokens, completion_tokens
							)
					else:
						done = getattr(chunk, 'done', False)
						msg = getattr(chunk, 'message', None)
						content = getattr(msg, 'content', '') if msg else ''
						if done:
							prompt_tokens = getattr(chunk, 'prompt_eval_count', 0) or 0
							completion_tokens = getattr(chunk, 'eval_count', 0) or 0
							ModelUsage.get_instance().record(
								self.config.id, prompt_tokens, completion_tokens
							)
					if content and not done:
						yield content

			return _gen()

		# Call Ollama chat API
		response = await self._client.chat(
			model=self.config.id,
			messages=messages,
			options=options if options else None,
			**kwargs
		)

		self._record_usage(response)
		return self._extract_content(response)

	def _record_usage(self, response: Any) -> None:
		'''Extract usage from Ollama response and record to ModelUsage.'''
		if isinstance(response, dict):
			prompt_tokens = response.get('prompt_eval_count', 0) or 0
			completion_tokens = response.get('eval_count', 0) or 0
		else:
			prompt_tokens = getattr(response, 'prompt_eval_count', 0) or 0
			completion_tokens = getattr(response, 'eval_count', 0) or 0
		ModelUsage.get_instance().record(self.config.id, prompt_tokens, completion_tokens)

	@staticmethod
	def _extract_content(response: Any) -> str:
		'''Extract content from Ollama ChatResponse.'''
		# Ollama SDK returns ChatResponse object with message.content attribute
		if hasattr(response, 'message'):
			message = response.message
			if hasattr(message, 'content') and message.content:
				return str(message.content)

		# Fallback for dict format
		if isinstance(response, dict):
			if 'message' in response:
				msg = response['message']
				if isinstance(msg, dict) and 'content' in msg:
					return str(msg['content'])
			if 'content' in response:
				return str(response['content'])

		raise ValueError('Ollama response missing generated content')
