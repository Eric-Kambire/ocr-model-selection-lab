"""Core services for the OCR benchmarking application."""

from .domain import BenchmarkCase, BenchmarkResult, InferenceResult, InferenceStatus
from .registry import ModelRegistry, build_default_registry
from .runner import BenchmarkRunner

__all__ = [
    "BenchmarkCase",
    "BenchmarkResult",
    "BenchmarkRunner",
    "InferenceResult",
    "InferenceStatus",
    "ModelRegistry",
    "build_default_registry",
]
