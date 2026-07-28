"""Tests du contrat de sortie structuré, indépendants d'Ollama."""

from ocr_benchmark.runner import BenchmarkRunner


class _StructuredAdapter:
    def __init__(self) -> None:
        self.received = None

    def perform_ocr(
        self,
        image_path,
        *,
        prompt=None,
        system_prompt=None,
        output_format="prompt",
        output_schema=None,
    ):
        self.received = {
            "image_path": image_path,
            "prompt": prompt,
            "system_prompt": system_prompt,
            "output_format": output_format,
            "output_schema": output_schema,
        }
        return {"text": '{"cin": null}', "latency": 0.01, "status": "success"}


class _LegacyAdapter:
    def perform_ocr(self, image_path, *, prompt=None):
        return {"text": "legacy", "latency": 0.01, "status": "success"}


def test_runner_forwards_the_schema_only_to_capable_adapters():
    adapter = _StructuredAdapter()
    schema = {"type": "object", "properties": {"cin": {"type": ["string", "null"]}}}

    BenchmarkRunner._perform_with_timeout(
        adapter,
        "image.png",
        None,
        prompt="Return JSON",
        system_prompt="System",
        output_format="schema",
        output_schema=schema,
    )

    assert adapter.received["output_format"] == "schema"
    assert adapter.received["output_schema"] == schema


def test_legacy_adapters_remain_compatible():
    result = BenchmarkRunner._perform_with_timeout(
        _LegacyAdapter(),
        "image.png",
        None,
        prompt="OCR",
        output_format="schema",
        output_schema={"type": "object"},
    )

    assert result["text"] == "legacy"
