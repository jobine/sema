from typing import Any, Dict
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import AsyncOpenAI
from .base_model import LLMConfig, AsyncBaseLLM


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
	async def __call__(self, prompt: str, **kwargs: Any) -> str:
		'''Call the LLM asynchronously via OpenAI Chat Completions API and return text.'''
		payload: Dict[str, Any] = {
			'model': self.config.name,
			'messages': [{'role': 'user', 'content': prompt}],
			'temperature': self.config.temperature,
		}
		
		payload_override: Dict[str, Any] = kwargs.pop('payload_override', {})
		payload.update(payload_override)

		response = await self._client.chat.completions.create(**payload)
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