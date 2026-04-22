import pytest
from unittest.mock import AsyncMock, MagicMock
from src.models.model_usage import ModelUsage


class TestOpenAIUsage:
    def setup_method(self):
        ModelUsage._instance = None

    @pytest.mark.asyncio
    async def test_non_streaming_records_usage(self, tmp_path):
        from src.models.openai_model import AsyncOpenAILLM
        from src.models.base import LLMConfig

        cache_file = tmp_path / "prices.json"
        cache_file.write_text("{}", encoding="utf-8")
        usage = ModelUsage.get_instance(pricing_path=cache_file)

        config = LLMConfig(
            id="gpt-4o-mini", provider="openai", description="",
            base_url="", api_key="fake", temperature=0.7,
        )

        # Mock response with usage
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 42
        mock_usage.completion_tokens = 18

        mock_message = MagicMock()
        mock_message.content = "Hello world"

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        llm = AsyncOpenAILLM(config, client=mock_client)
        result = await llm("test prompt")

        assert result == "Hello world"
        assert usage.total_prompt_tokens == 42
        assert usage.total_completion_tokens == 18

    @pytest.mark.asyncio
    async def test_streaming_records_usage(self, tmp_path):
        from src.models.openai_model import AsyncOpenAILLM
        from src.models.base import LLMConfig

        cache_file = tmp_path / "prices.json"
        cache_file.write_text("{}", encoding="utf-8")
        usage = ModelUsage.get_instance(pricing_path=cache_file)

        config = LLMConfig(
            id="gpt-4o-mini", provider="openai", description="",
            base_url="", api_key="fake", temperature=0.7,
        )

        chunk1 = MagicMock()
        chunk1.usage = None
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = "Hello"

        chunk2 = MagicMock()
        chunk2.usage = None
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = " world"

        final_chunk = MagicMock()
        final_usage = MagicMock()
        final_usage.prompt_tokens = 10
        final_usage.completion_tokens = 5
        final_chunk.usage = final_usage
        final_chunk.choices = []

        async def mock_stream():
            for c in [chunk1, chunk2, final_chunk]:
                yield c

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())

        llm = AsyncOpenAILLM(config, client=mock_client)
        result_iter = await llm("test", payload_override={"stream": True})

        chunks = []
        async for text in result_iter:
            chunks.append(text)

        assert chunks == ["Hello", " world"]
        assert usage.total_prompt_tokens == 10
        assert usage.total_completion_tokens == 5
