from collections.abc import AsyncIterator
from typing import Any, Dict
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google import genai
from .base import LLMConfig, AsyncBaseLLM
from .model_usage import ModelUsage


class AsyncGeminiLLM(AsyncBaseLLM):
	'''Async wrapper for Google Gemini API.'''

	def __init__(
		self,
		config: LLMConfig,
		*,
		client: genai.Client | None = None,
	) -> None:
		super().__init__(config)
		self._client = client or genai.Client(api_key=self.config.api_key or None)

	@retry(
		reraise=True,
		stop=stop_after_attempt(3),
		wait=wait_exponential(multiplier=1, min=1, max=10),
		retry=retry_if_exception_type(Exception),
	)
	async def __call__(self, prompt: str, **kwargs: Any) -> str | AsyncIterator[str]:
		'''Call the Gemini LLM. Returns str or AsyncIterator[str] for streaming.'''
		payload: Dict[str, Any] = {
			'model': self.config.id,
			'contents': prompt,
		}

		config_dict: Dict[str, Any] = {}
		if self.config.temperature is not None:
			config_dict['temperature'] = self.config.temperature

		payload_override: Dict[str, Any] = kwargs.pop('payload_override', {})
		config_dict.update(payload_override.pop('config', {}))

		if config_dict:
			payload['config'] = genai.types.GenerateContentConfig(**config_dict)

		stream = payload_override.pop('stream', False)
		payload.update(payload_override)

		if stream:
			async def _gen() -> AsyncIterator[str]:
				prompt_tokens = 0
				completion_tokens = 0
				async with self._client.aio as aio_client:
					async for chunk in await aio_client.models.generate_content_stream(**payload):
						um = getattr(chunk, 'usage_metadata', None)
						if um:
							prompt_tokens = getattr(um, 'prompt_token_count', 0) or 0
							completion_tokens = getattr(um, 'candidates_token_count', 0) or 0
						text = getattr(chunk, 'text', None)
						if text:
							yield text
				ModelUsage.get_instance().record(self.config.id, prompt_tokens, completion_tokens)

			return _gen()

		async with self._client.aio as aio_client:
			response = await aio_client.models.generate_content(**payload)

		self._record_usage(response)
		return self._extract_content(response)

	def _record_usage(self, response: Any) -> None:
		'''Extract usage from Gemini response and record to ModelUsage.'''
		um = getattr(response, 'usage_metadata', None)
		if um is None:
			return
		prompt_tokens = getattr(um, 'prompt_token_count', 0) or 0
		completion_tokens = getattr(um, 'candidates_token_count', 0) or 0
		ModelUsage.get_instance().record(self.config.id, prompt_tokens, completion_tokens)

	@staticmethod
	def _extract_content(response: Any) -> str:
		if hasattr(response, 'text') and response.text:
			return str(response.text)

		if hasattr(response, 'candidates') and response.candidates:
			candidate = response.candidates[0]
			if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
				parts = candidate.content.parts
				if parts and hasattr(parts[0], 'text'):
					return str(parts[0].text)

		raise ValueError('Gemini response missing generated content')
