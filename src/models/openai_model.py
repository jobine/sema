from collections.abc import AsyncIterator
from typing import Any, Dict
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import AsyncOpenAI
from .base import LLMConfig, AsyncBaseLLM
from .model_usage import ModelUsage


class AsyncOpenAILLM(AsyncBaseLLM):
	'''Async wrapper for OpenAI-compatible APIs.'''

	def __init__(
		self,
		config: LLMConfig,
		*,
		client: AsyncOpenAI | None = None,
	) -> None:
		super().__init__(config)
		self._client = client or AsyncOpenAI(
			api_key=self.config.api_key or None,
			base_url=self.config.base_url or None,
		)

	@retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10), retry=retry_if_exception_type(Exception))
	async def __call__(self, prompt: str, **kwargs: Any) -> str | AsyncIterator[str]:
		'''Call the LLM. Returns str (non-streaming) or AsyncIterator[str] (streaming).'''
		payload: Dict[str, Any] = {
			'model': self.config.id,
			'messages': [{'role': 'user', 'content': prompt}],
			'temperature': self.config.temperature,
		}

		payload_override: Dict[str, Any] = kwargs.pop('payload_override', {})
		payload.update(payload_override)

		if payload.get('stream'):
			stream_payload = {**payload, 'stream_options': {'include_usage': True}}

			async def _gen() -> AsyncIterator[str]:
				stream = await self._client.chat.completions.create(**stream_payload)
				async for chunk in stream:
					# OpenAI spec: usage appears only on the final chunk with empty choices.
					# Guard against servers that emit usage on multiple chunks.
					if chunk.usage and not chunk.choices:
						self._record_usage(chunk)
					if chunk.choices and chunk.choices[0].delta.content:
						yield chunk.choices[0].delta.content

			return _gen()

		response = await self._client.chat.completions.create(**payload)
		self._record_usage(response)
		return self._extract_content(response)

	def _record_usage(self, response: Any) -> None:
		'''Extract usage from response and record to ModelUsage.'''
		usage = getattr(response, 'usage', None)
		if usage is None:
			return
		prompt_tokens = getattr(usage, 'prompt_tokens', 0) or 0
		completion_tokens = getattr(usage, 'completion_tokens', 0) or 0
		ModelUsage.get_instance().record(self.config.id, prompt_tokens, completion_tokens)

	@staticmethod
	def _extract_content(response: Any) -> str:
		choices = getattr(response, 'choices', None)
		if choices is None and isinstance(response, dict):
			choices = response.get('choices')

		if choices:
			first = choices[0]
			message = getattr(first, 'message', None) or (first.get('message') if isinstance(first, dict) else None)
			if message:
				content = getattr(message, 'content', None) or (message.get('content') if isinstance(message, dict) else None)
				if content is not None:
					return str(content)

			text = getattr(first, 'text', None) or (first.get('text') if isinstance(first, dict) else None)
			if text is not None:
				return str(text)

		raise ValueError('LLM response missing generated content')
