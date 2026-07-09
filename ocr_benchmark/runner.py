from __future__ import annotations

import math
import time
import uuid
from collections.abc import Callable, Iterable
from typing import Any

import pandas as pd

from evaluator import evaluate_bankmark, evaluate_ocr

from .domain import BenchmarkCase, BenchmarkResult, InferenceResult, InferenceStatus
from .registry import ModelRegistry

ProgressCallback = Callable[[int, int, str], None]


class BenchmarkRunner:
    """Runs models against immutable cases and produces normalized results."""

    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry

    def run(
        self,
        model_specs: Iterable[str],
        cases: Iterable[BenchmarkCase],
        *,
        eval_mode: str = "Standard",
        mock_noise: float = 0.05,
        progress: ProgressCallback | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        selected_cases = list(cases)
        models = [
            self.registry.create(spec, mock_noise=mock_noise)
            for spec in model_specs
        ]
        run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        total = len(models) * len(selected_cases)
        completed = 0
        results: list[dict[str, Any]] = []

        for model in models:
            for case in selected_cases:
                completed += 1
                if progress:
                    progress(completed, total, f"{model.model_name}: {case.image_path}")
                try:
                    raw = model.perform_ocr(case.image_path)
                    inference = (
                        raw
                        if isinstance(raw, InferenceResult)
                        else InferenceResult.from_legacy_dict(raw)
                    )
                except Exception as exc:  # Adapter boundary: keep one failure local.
                    inference = InferenceResult(
                        text="",
                        latency_seconds=0.0,
                        status=InferenceStatus.FAILED,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                result = self._evaluate(
                    run_id, model.model_name, case, inference, eval_mode
                )
                results.append(result.to_dict())

        return run_id, results

    @staticmethod
    def _evaluate(
        run_id: str,
        model_name: str,
        case: BenchmarkCase,
        inference: InferenceResult,
        eval_mode: str,
    ) -> BenchmarkResult:
        common = dict(
            run_id=run_id,
            model=model_name,
            image_path=case.image_path,
            category=case.category,
            description=case.description,
            ground_truth=case.ground_truth,
            extracted_text=inference.text,
            latency=inference.latency_seconds,
            status=inference.status.value,
            error=inference.error,
            eval_mode=eval_mode,
            device=inference.device,
            input_tokens=inference.input_tokens,
            output_tokens=inference.output_tokens,
            tokens_per_second=inference.tokens_per_second,
        )
        if inference.status is not InferenceStatus.SUCCESS:
            return BenchmarkResult(accuracy=None, cer=None, wer=None, **common)

        if eval_mode == "Bankmark":
            metrics = evaluate_bankmark(case.ground_truth, inference.text)
            return BenchmarkResult(
                accuracy=metrics["bankmark_score"],
                cer=metrics["numeric_cer"],
                wer=metrics["general_wer"],
                iban_emr=metrics["iban_emr"],
                iban_valid_rate=metrics["iban_valid_rate"],
                iban_status=metrics["iban_status"],
                amount_emr=metrics["amount_emr"],
                amount_status=metrics["amount_status"],
                **common,
            )

        metrics = evaluate_ocr(case.ground_truth, inference.text)
        structure = metrics["structure"]
        return BenchmarkResult(
            accuracy=metrics["accuracy_score"],
            cer=metrics["cer_normalized"],
            wer=metrics["wer_normalized"],
            table_score=structure["table_preservation_score"],
            table_status=structure["table_status"],
            math_score=structure["math_preservation_score"],
            math_status=structure["math_status"],
            **common,
        )


def summarize_results(results: list[dict[str, Any]]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    rows: list[dict[str, Any]] = []
    for model, group in df.groupby("model", sort=False):
        successful = group[group["status"] == "success"]
        latencies = successful["latency"].dropna()
        quality = successful["accuracy"].dropna()
        token_speed = successful["tokens_per_second"].dropna()
        row = {
            "Model": model,
            "Device": _unique_join(group["device"]),
            "Documents": len(group),
            "Success rate": len(successful) / len(group),
            "Quality score": quality.mean() if not quality.empty else math.nan,
            "Mean latency (s)": latencies.mean() if not latencies.empty else math.nan,
            "Median latency (s)": latencies.median() if not latencies.empty else math.nan,
            "P95 latency (s)": latencies.quantile(0.95) if not latencies.empty else math.nan,
            "Documents/s": (1.0 / latencies.mean()) if not latencies.empty and latencies.mean() else math.nan,
            "Tokens/s": token_speed.mean() if not token_speed.empty else math.nan,
            "Output tokens": successful["output_tokens"].sum(min_count=1),
            "CER": successful["cer"].mean(),
            "WER": successful["wer"].mean(),
        }
        if "table_score" in successful:
            row["Table preservation"] = successful["table_score"].mean()
        if "math_score" in successful:
            row["Math preservation"] = successful["math_score"].mean()
        if "iban_emr" in successful:
            row["IBAN exact match"] = successful["iban_emr"].mean()
        if "amount_emr" in successful:
            row["Amount exact match"] = successful["amount_emr"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def _unique_join(values: pd.Series) -> str:
    return ", ".join(sorted({str(value) for value in values if value}))
