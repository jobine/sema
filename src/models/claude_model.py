from typing import Any, Dict
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from anthropic import AsyncAnthropic
from .base_model import LLMConfig, AsyncBaseLLM


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
	async def __call__(self, prompt: str, **kwargs: Any) -> str:
		'''Call the Anthropic Claude LLM asynchronously via Messages API and return text.'''
		payload: Dict[str, Any] = {
			'model': self.config.name,
			'messages': [{'role': 'user', 'content': prompt}],
			'max_tokens': 4096,
		}
		# Add optional parameters if configured
		if self.config.temperature is not None:
			payload['temperature'] = self.config.temperature

		payload_override: Dict[str, Any] = kwargs.pop('payload_override', {})
		payload.update(payload_override)

		response = await self._client.messages.create(**payload)
		return self._extract_content(response)

	@staticmethod
	def _extract_content(response: Any) -> str:
		# Handle Anthropic Message response structure
		# Response has 'content' which is a list of content blocks
		content = getattr(response, 'content', None)
		if content is None and isinstance(response, dict):
			content = response.get('content')

		if content:
			# Extract text from content blocks
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