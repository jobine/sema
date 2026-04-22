import json
import pytest
from pathlib import Path
from unittest.mock import patch
from src.models.model_usage import ModelUsage


class TestModelUsage:
    def setup_method(self):
        """Reset singleton before each test."""
        ModelUsage._instance = None

    def test_load_prices_from_local_cache(self, tmp_path):
        prices = {
            "gpt-4o-mini": {
                "input_cost_per_token": 0.00000015,
                "output_cost_per_token": 0.0000006,
            }
        }
        cache_file = tmp_path / "model_prices.json"
        cache_file.write_text(json.dumps(prices), encoding="utf-8")

        usage = ModelUsage(pricing_path=cache_file)
        assert usage._prices["gpt-4o-mini"]["input_cost_per_token"] == 0.00000015
        assert usage._prices["gpt-4o-mini"]["output_cost_per_token"] == 0.0000006

    def test_record_accumulates_tokens_and_cost(self, tmp_path):
        prices = {
            "gpt-4o-mini": {
                "input_cost_per_token": 0.00000015,
                "output_cost_per_token": 0.0000006,
            }
        }
        cache_file = tmp_path / "model_prices.json"
        cache_file.write_text(json.dumps(prices), encoding="utf-8")

        usage = ModelUsage(pricing_path=cache_file)
        usage.record("gpt-4o-mini", prompt_tokens=100, completion_tokens=50)
        usage.record("gpt-4o-mini", prompt_tokens=200, completion_tokens=100)

        assert usage.total_prompt_tokens == 300
        assert usage.total_completion_tokens == 150
        expected_cost = (100 * 0.00000015 + 50 * 0.0000006) + (200 * 0.00000015 + 100 * 0.0000006)
        assert abs(usage.total_cost - expected_cost) < 1e-12

    def test_record_unknown_model_zero_cost(self, tmp_path):
        cache_file = tmp_path / "model_prices.json"
        cache_file.write_text("{}", encoding="utf-8")

        usage = ModelUsage(pricing_path=cache_file)
        usage.record("unknown-model", prompt_tokens=500, completion_tokens=200)

        assert usage.total_prompt_tokens == 500
        assert usage.total_completion_tokens == 200
        assert usage.total_cost == 0.0

    def test_summary_format(self, tmp_path):
        cache_file = tmp_path / "model_prices.json"
        cache_file.write_text("{}", encoding="utf-8")

        usage = ModelUsage(pricing_path=cache_file)
        usage.record("x", prompt_tokens=1000, completion_tokens=500)

        s = usage.summary()
        assert "1,000" in s
        assert "500" in s

    def test_report_section_markdown(self, tmp_path):
        cache_file = tmp_path / "model_prices.json"
        cache_file.write_text("{}", encoding="utf-8")

        usage = ModelUsage(pricing_path=cache_file)
        usage.record("x", prompt_tokens=1234, completion_tokens=567)

        md = usage.report_section()
        assert "## Model Usage" in md
        assert "1,234" in md
        assert "567" in md

    def test_get_instance_singleton(self, tmp_path):
        cache_file = tmp_path / "model_prices.json"
        cache_file.write_text("{}", encoding="utf-8")

        a = ModelUsage.get_instance(pricing_path=cache_file)
        b = ModelUsage.get_instance()
        assert a is b

    def test_missing_pricing_file_no_crash(self, tmp_path):
        usage = ModelUsage(pricing_path=tmp_path / "nonexistent.json")
        usage.record("gpt-4", prompt_tokens=10, completion_tokens=5)
        assert usage.total_prompt_tokens == 10
        assert usage.total_cost == 0.0
