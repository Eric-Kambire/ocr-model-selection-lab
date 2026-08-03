"""Tests de non-régression contre les faux succès des modèles textuels."""

from __future__ import annotations

import sys
from pathlib import Path

from models.ollama_capabilities import inspect_ollama_vision_capability
from models.ollama_model import OllamaOCRModel


class _FakeOllamaClient:
    def __init__(self, capabilities: list[str]) -> None:
        self.capabilities = capabilities
        self.chat_calls: list[dict] = []
        self.configured_host: str | None = None
        self.configured_timeout: float | None = None

    def Client(self, *, host: str, timeout: float):
        self.configured_host = host
        self.configured_timeout = timeout
        return self

    def show(self, *, model: str) -> dict:
        return {"capabilities": self.capabilities, "model": model}

    def chat(self, **request):
        self.chat_calls.append(request)
        return {
            "message": {"content": '{"cin":"INVENTE"}'},
            "prompt_eval_count": 10,
            "eval_count": 4,
            "eval_duration": 1_000_000_000,
        }

    def generate(self, **_request):
        return {}


def _build_model(monkeypatch, client: _FakeOllamaClient) -> OllamaOCRModel:
    monkeypatch.setitem(sys.modules, "ollama", client)
    return OllamaOCRModel(
        "modele-test",
        unload_after_task=False,
        request_timeout=123,
    )


def test_text_model_is_rejected_before_chat(monkeypatch, tmp_path: Path):
    image = tmp_path / "recto.png"
    image.write_bytes(b"image")
    client = _FakeOllamaClient(["completion"])
    model = _build_model(monkeypatch, client)

    result = model.perform_ocr(str(image), prompt="Retourne un JSON.")

    assert result["status"] == "incompatible_model"
    assert result["image_submitted"] is False
    assert result["text"] == ""
    assert "Aucune image n'a été analysée" in result["error"]
    assert client.chat_calls == []
    assert client.configured_timeout == 123


def test_vision_model_sends_the_image_and_can_succeed(monkeypatch, tmp_path: Path):
    image = tmp_path / "recto.png"
    image.write_bytes(b"image")
    client = _FakeOllamaClient(["completion", "vision"])
    model = _build_model(monkeypatch, client)

    result = model.perform_ocr(str(image), prompt="Retourne un JSON.")

    assert result["status"] == "success"
    assert result["image_submitted"] is True
    assert client.chat_calls[0]["messages"][-1]["images"] == [str(image)]
    assert result["configured_timeout_seconds"] == 123


def test_multimodal_metadata_is_supported_for_older_show_response():
    class _LegacyClient:
        @staticmethod
        def show(*, model: str) -> dict:
            return {
                "model": model,
                "model_info": {"example.mm.tokens_per_image": 256},
            }

    result = inspect_ollama_vision_capability(_LegacyClient(), "ancien-vlm")

    assert result.supported is True
    assert result.capabilities == ("vision",)
