import asyncio
from typing import Any, Dict
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from zai import ZhipuAiClient
from .base import LLMConfig, AsyncBaseLLM


class AsyncZhipuLLM(AsyncBaseLLM):
	'''Async wrapper for Zhipu GLM API (zai-sdk).'''

	def __init__(
		self,
		config: LLMConfig,
		*,
		client: ZhipuAiClient | None = None,
	) -> None:
		super().__init__(config)
		self._client = client or ZhipuAiClient(api_key=self.config.api_key or None)

	@retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10), retry=retry_if_exception_type(Exception))
	async def __call__(self, prompt: str, **kwargs: Any) -> str:
		'''Call the Zhipu GLM LLM asynchronously and return text.'''
		payload: Dict[str, Any] = {
			'model': self.config.name,
			'messages': [{'role': 'user', 'content': prompt}],
		}
		if self.config.temperature is not None:
			payload['temperature'] = self.config.temperature

		payload_override: Dict[str, Any] = kwargs.pop('payload_override', {})
		payload.update(payload_override)

		response = await asyncio.to_thread(
			self._client.chat.completions.create, **payload
		)
		return self._extract_content(response)

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
