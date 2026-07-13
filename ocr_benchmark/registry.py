from __future__ import annotations

from collections.abc import Callable
import logging
from typing import Any

from models.mock_model import MockOCRModel
from models.ollama_model import OllamaOCRModel

LOGGER = logging.getLogger(__name__)

ModelFactory = Callable[..., Any]


class ModelRegistry:
    """Maps stable provider names to OCR adapter factories."""

    def __init__(self) -> None:
        self._factories: dict[str, ModelFactory] = {}

    def register(self, provider: str, factory: ModelFactory) -> None:
        key = provider.strip().lower()
        if not key or ":" in key:
            raise ValueError("Provider names must be non-empty and cannot contain ':'.")
        if key in self._factories:
            raise ValueError(f"Provider '{key}' is already registered.")
        self._factories[key] = factory

    def create(self, model_spec: str, **options: Any) -> Any:
        if ":" in model_spec:
            provider, model_name = model_spec.split(":", 1)
        elif model_spec.startswith("Mock"):
            provider, model_name = "mock", model_spec
        elif model_spec == "EasyOCR-Local":
            provider, model_name = "easyocr", model_spec
        else:
            # Backward compatibility: historical names are Ollama model names.
            provider, model_name = "ollama", model_spec

        factory = self._factories.get(provider.lower())
        if factory is None:
            available = ", ".join(sorted(self._factories))
            raise ValueError(f"Unknown provider '{provider}'. Available providers: {available}")
        LOGGER.info("Creating model adapter | spec=%s | provider=%s | model=%s", model_spec, provider, model_name)
        return factory(model_name=model_name, **options)

    @property
    def providers(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


def build_default_registry() -> ModelRegistry:
    registry = ModelRegistry()
    registry.register(
        "mock",
        lambda model_name, mock_noise=0.05, **_: MockOCRModel(
            model_name=model_name,
            error_rate=mock_noise,
        ),
    )
    registry.register(
        "ollama",
        lambda model_name, model_prompt=None, cpu_threads=None, unload_after_task=False, timeout_seconds=None, **_: OllamaOCRModel(
            model_name=model_name,
            prompt=model_prompt,
            cpu_threads=cpu_threads,
            unload_after_task=unload_after_task,
            request_timeout=timeout_seconds,
        ),
    )

    try:
        from models.easyocr_model import EASYOCR_AVAILABLE, EasyOCRModel

        if EASYOCR_AVAILABLE:
            registry.register("easyocr", lambda model_name, **_: EasyOCRModel())
    except ImportError:
        pass

    return registry
