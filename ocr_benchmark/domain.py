from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class InferenceStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class BenchmarkCase:
    image_path: str
    ground_truth: str
    category: str
    description: str = ""
    prompt: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BenchmarkCase":
        return cls(
            image_path=str(value["image_path"]),
            ground_truth=str(value["ground_truth"]),
            category=str(value["category"]),
            description=str(value.get("description", "")),
            prompt=(str(value["prompt"]) if value.get("prompt") is not None else None),
        )


@dataclass(frozen=True)
class InferenceResult:
    text: str
    latency_seconds: float
    status: InferenceStatus = InferenceStatus.SUCCESS
    error: str | None = None
    raw_response: str | None = None
    reasoning: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    tokens_per_second: float | None = None
    device: str = "unknown"

    @classmethod
    def from_legacy_dict(cls, value: dict[str, Any]) -> "InferenceResult":
        error = value.get("error")
        status_value = value.get("status", "failed" if error else "success")
        try:
            status = InferenceStatus(status_value)
        except ValueError:
            status = InferenceStatus.FAILED
        return cls(
            text=str(value.get("text", "")),
            latency_seconds=float(value.get("latency", 0.0)),
            status=status,
            error=str(error) if error else None,
            raw_response=value.get("raw_response"),
            reasoning=value.get("reasoning") or value.get("thinking"),
            input_tokens=_optional_int(value.get("input_tokens")),
            output_tokens=_optional_int(value.get("output_tokens")),
            tokens_per_second=_optional_float(value.get("tokens_per_second")),
            device=str(value.get("device", "unknown")),
        )


@dataclass(frozen=True)
class BenchmarkResult:
    run_id: str
    model: str
    image_path: str
    category: str
    description: str
    ground_truth: str
    extracted_text: str
    latency: float
    status: str
    error: str | None
    eval_mode: str
    accuracy: float | None
    cer: float | None
    wer: float | None
    device: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    tokens_per_second: float | None = None
    raw_response: str | None = None
    reasoning: str | None = None
    table_score: float | None = None
    table_status: str | None = None
    math_score: float | None = None
    math_status: str | None = None
    iban_emr: float | None = None
    iban_valid_rate: float | None = None
    iban_status: str | None = None
    amount_emr: float | None = None
    amount_status: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None
