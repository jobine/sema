from typing import Any, Dict
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google import genai
from .base_model import LLMConfig, AsyncBaseLLM


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
	async def __call__(self, prompt: str, **kwargs: Any) -> str:
		'''Call the Gemini LLM asynchronously and return text.'''
		payload: Dict[str, Any] = {
			'model': self.config.name,
			'contents': prompt,
		}

		# Build generation config
		config_dict: Dict[str, Any] = {}
		if self.config.temperature is not None:
			config_dict['temperature'] = self.config.temperature

		payload_override: Dict[str, Any] = kwargs.pop('payload_override', {})
		config_dict.update(payload_override.pop('config', {}))

		if config_dict:
			payload['config'] = genai.types.GenerateContentConfig(**config_dict)

		payload.update(payload_override)

		async with self._client.aio as aio_client:
			response = await aio_client.models.generate_content(**payload)

		return self._extract_content(response)

	@staticmethod
	def _extract_content(response: Any) -> str:
		# Handle Gemini response structure
		if hasattr(response, 'text') and response.text:
			return str(response.text)

		if hasattr(response, 'candidates') and response.candidates:
			candidate = response.candidates[0]
			if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
				parts = candidate.content.parts
				if parts and hasattr(parts[0], 'text'):
					return str(parts[0].text)

		raise ValueError('Gemini response missing generated content')