from collections.abc import AsyncIterator
from typing import Any, Dict
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from anthropic import AsyncAnthropic
from .base import LLMConfig, AsyncBaseLLM
from .model_usage import ModelUsage


class AsyncAnthropicLLM(AsyncBaseLLM):
	'''Async wrapper for Anthropic Claude API.'''

	def __init__(
		self,
		config: LLMConfig,
		*,
		client: AsyncAnthropic | None = None,
	) -> None:
		super().__init__(config)
		self._client = client or AsyncAnthropic(
			api_key=self.config.api_key or None,
			base_url=self.config.base_url or None,
		)

	@retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10), retry=retry_if_exception_type(Exception))
	async def __call__(self, prompt: str, **kwargs: Any) -> str | AsyncIterator[str]:
		'''Call the Anthropic Claude LLM. Returns str or AsyncIterator[str] for streaming.'''
		payload: Dict[str, Any] = {
			'model': self.config.id,
			'messages': [{'role': 'user', 'content': prompt}],
			'max_tokens': 4096,
		}
		if self.config.temperature is not None:
			payload['temperature'] = self.config.temperature

		payload_override: Dict[str, Any] = kwargs.pop('payload_override', {})
		payload.update(payload_override)

		if payload.pop('stream', False):
			async def _gen() -> AsyncIterator[str]:
				async with self._client.messages.stream(**payload) as stream:
					async for text in stream.text_stream:
						yield text
					message = await stream.get_final_message()
					self._record_usage(message)

			return _gen()

		response = await self._client.messages.create(**payload)
		self._record_usage(response)
		return self._extract_content(response)

	def _record_usage(self, response: Any) -> None:
		'''Extract usage from Anthropic response and record to ModelUsage.'''
		usage = getattr(response, 'usage', None)
		if usage is None:
			return
		prompt_tokens = getattr(usage, 'input_tokens', 0) or 0
		completion_tokens = getattr(usage, 'output_tokens', 0) or 0
		ModelUsage.get_instance().record(self.config.id, prompt_tokens, completion_tokens)

	@staticmethod
	def _extract_content(response: Any) -> str:
		content = getattr(response, 'content', None)
		if content is None and isinstance(response, dict):
			content = response.get('content')

		if content:
			text_parts = []
			for block in content:
				block_type = getattr(block, 'type', None) or (block.get('type') if isinstance(block, dict) else None)
				if block_type == 'text':
					text = getattr(block, 'text', None) or (block.get('text') if isinstance(block, dict) else None)
					if text is not None:
						text_parts.append(str(text))
			if text_parts:
				return ''.join(text_parts)

		raise ValueError('LLM response missing generated content')
