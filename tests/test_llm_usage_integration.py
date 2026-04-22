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


class TestAnthropicUsage:
    def setup_method(self):
        ModelUsage._instance = None

    @pytest.mark.asyncio
    async def test_non_streaming_records_usage(self, tmp_path):
        from src.models.claude_model import AsyncAnthropicLLM
        from src.models.base import LLMConfig

        cache_file = tmp_path / "prices.json"
        cache_file.write_text("{}", encoding="utf-8")
        usage = ModelUsage.get_instance(pricing_path=cache_file)

        config = LLMConfig(
            id="claude-sonnet-4-20250514", provider="anthropic", description="",
            base_url="", api_key="fake", temperature=0.7,
        )

        mock_usage = MagicMock()
        mock_usage.input_tokens = 55
        mock_usage.output_tokens = 30

        mock_block = MagicMock()
        mock_block.type = "text"
        mock_block.text = "Answer"

        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_response.usage = mock_usage

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        llm = AsyncAnthropicLLM(config, client=mock_client)
        result = await llm("test prompt")

        assert result == "Answer"
        assert usage.total_prompt_tokens == 55
        assert usage.total_completion_tokens == 30


class TestGeminiUsage:
    def setup_method(self):
        ModelUsage._instance = None

    @pytest.mark.asyncio
    async def test_non_streaming_records_usage(self, tmp_path):
        from src.models.gemini_model import AsyncGeminiLLM
        from src.models.base import LLMConfig

        cache_file = tmp_path / "prices.json"
        cache_file.write_text("{}", encoding="utf-8")
        usage = ModelUsage.get_instance(pricing_path=cache_file)

        config = LLMConfig(
            id="gemini-2.5-flash", provider="gemini", description="",
            base_url="", api_key="fake", temperature=0.7,
        )

        mock_usage_metadata = MagicMock()
        mock_usage_metadata.prompt_token_count = 80
        mock_usage_metadata.candidates_token_count = 40

        mock_response = MagicMock()
        mock_response.text = "Gemini answer"
        mock_response.usage_metadata = mock_usage_metadata

        mock_aio = AsyncMock()
        mock_aio.models.generate_content = AsyncMock(return_value=mock_response)

        mock_client = MagicMock()
        mock_client.aio.__aenter__ = AsyncMock(return_value=mock_aio)
        mock_client.aio.__aexit__ = AsyncMock(return_value=False)

        llm = AsyncGeminiLLM(config, client=mock_client)
        result = await llm("test prompt")

        assert result == "Gemini answer"
        assert usage.total_prompt_tokens == 80
        assert usage.total_completion_tokens == 40


class TestZhipuUsage:
    def setup_method(self):
        ModelUsage._instance = None

    @pytest.mark.asyncio
    async def test_non_streaming_records_usage(self, tmp_path):
        from src.models.zhipu_model import AsyncZhipuLLM
        from src.models.base import LLMConfig

        cache_file = tmp_path / "prices.json"
        cache_file.write_text("{}", encoding="utf-8")
        usage = ModelUsage.get_instance(pricing_path=cache_file)

        config = LLMConfig(
            id="glm-4-flash", provider="zhipu", description="",
            base_url="", api_key="fake", temperature=0.7,
        )

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 60
        mock_usage.completion_tokens = 25

        mock_message = MagicMock()
        mock_message.content = "Zhipu answer"

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        mock_client = MagicMock()
        mock_client.chat.completions.create = MagicMock(return_value=mock_response)

        llm = AsyncZhipuLLM(config, client=mock_client)
        result = await llm("test prompt")

        assert result == "Zhipu answer"
        assert usage.total_prompt_tokens == 60
        assert usage.total_completion_tokens == 25


class TestOllamaUsage:
    def setup_method(self):
        ModelUsage._instance = None

    @pytest.mark.asyncio
    async def test_non_streaming_records_usage(self, tmp_path):
        from src.models.ollama_model import AsyncOllamaLLM
        from src.models.base import LLMConfig

        cache_file = tmp_path / "prices.json"
        cache_file.write_text("{}", encoding="utf-8")
        usage = ModelUsage.get_instance(pricing_path=cache_file)

        config = LLMConfig(
            id="qwen2.5:7b", provider="ollama", description="",
            base_url="http://localhost:11434", api_key="", temperature=0.7,
        )

        mock_message = MagicMock()
        mock_message.content = "Ollama answer"

        mock_response = MagicMock()
        mock_response.message = mock_message
        mock_response.prompt_eval_count = 35
        mock_response.eval_count = 20

        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value=mock_response)

        llm = AsyncOllamaLLM(config, client=mock_client)
        result = await llm("test prompt")

        assert result == "Ollama answer"
        assert usage.total_prompt_tokens == 35
        assert usage.total_completion_tokens == 20


class TestMLXUsage:
    def setup_method(self):
        ModelUsage._instance = None

    @pytest.mark.asyncio
    async def test_non_streaming_records_usage(self, tmp_path):
        from src.models.mlx_model import AsyncMLXLLM
        from src.models.base import LLMConfig

        cache_file = tmp_path / "prices.json"
        cache_file.write_text("{}", encoding="utf-8")
        usage = ModelUsage.get_instance(pricing_path=cache_file)

        config = LLMConfig(
            id="mlx-community/gemma-2-2b", provider="mlx", description="",
            base_url="http://localhost:8080/v1", api_key="mlx", temperature=0.7,
        )

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 20
        mock_usage.completion_tokens = 10

        mock_message = MagicMock()
        mock_message.content = "MLX answer"

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        llm = AsyncMLXLLM(config, client=mock_client)
        result = await llm("test prompt")

        assert result == "MLX answer"
        assert usage.total_prompt_tokens == 20
        assert usage.total_completion_tokens == 10

    @pytest.mark.asyncio
    async def test_non_streaming_no_usage_no_crash(self, tmp_path):
        """MLX server may not return usage — should not crash."""
        from src.models.mlx_model import AsyncMLXLLM
        from src.models.base import LLMConfig

        cache_file = tmp_path / "prices.json"
        cache_file.write_text("{}", encoding="utf-8")
        usage = ModelUsage.get_instance(pricing_path=cache_file)

        config = LLMConfig(
            id="mlx-model", provider="mlx", description="",
            base_url="http://localhost:8080/v1", api_key="mlx", temperature=0.7,
        )

        mock_message = MagicMock()
        mock_message.content = "MLX answer"

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = None

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        llm = AsyncMLXLLM(config, client=mock_client)
        result = await llm("test prompt")

        assert result == "MLX answer"
        assert usage.total_prompt_tokens == 0
