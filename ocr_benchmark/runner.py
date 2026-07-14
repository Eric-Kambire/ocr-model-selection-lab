from __future__ import annotations

import math
import logging
import threading
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from evaluator import evaluate_bankmark, evaluate_ocr

from .domain import BenchmarkCase, BenchmarkResult, InferenceResult, InferenceStatus
from .registry import ModelRegistry

LOGGER = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], None]
TraceCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class RunnerProgress:
    run_id: str
    completed: int
    total: int
    model_name: str
    case: BenchmarkCase
    result: dict[str, Any] | None
    stage: str
    elapsed_seconds: float
    estimated_remaining_seconds: float | None
    error_count: int


class BenchmarkRunner:
    """Execute a benchmark while isolating provider failures.

    The runner intentionally owns the lifecycle of one adapter at a time:
    create -> process every selected case -> close.  This is important for
    local machines with limited RAM/VRAM and also prevents one broken provider
    from aborting the remaining models.
    """

    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry

    def run(
        self,
        model_specs: Iterable[str],
        cases: Iterable[BenchmarkCase],
        *,
        eval_mode: str = "Standard",
        mock_noise: float = 0.05,
        timeout_seconds: float | None = None,
        max_errors: int | None = None,
        model_prompt: str | None = None,
        cpu_threads: int | None = None,
        unload_after_task: bool = False,
        progress: ProgressCallback | None = None,
        trace: TraceCallback | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        updates = self.iter_run(
            model_specs,
            cases,
            eval_mode=eval_mode,
            mock_noise=mock_noise,
            timeout_seconds=timeout_seconds,
            max_errors=max_errors,
            model_prompt=model_prompt,
            cpu_threads=cpu_threads,
            unload_after_task=unload_after_task,
            progress=progress,
            trace=trace,
        )
        run_id = ""
        results: list[dict[str, Any]] = []
        for update in updates:
            run_id = update.run_id
            if update.stage == "completed" and update.result is not None:
                results.append(update.result)
        return run_id, results

    def iter_run(
        self,
        model_specs: Iterable[str],
        cases: Iterable[BenchmarkCase],
        *,
        eval_mode: str = "Standard",
        mock_noise: float = 0.05,
        timeout_seconds: float | None = None,
        max_errors: int | None = None,
        model_prompt: str | None = None,
        cpu_threads: int | None = None,
        unload_after_task: bool = False,
        progress: ProgressCallback | None = None,
        trace: TraceCallback | None = None,
    ):
        selected_cases = list(cases)
        model_specs = list(model_specs)
        LOGGER.info(
            "Runner initialised | models=%s | cases=%d | timeout=%s | cpu_threads=%s | unload_after_task=%s | eval_mode=%s",
            model_specs, len(selected_cases), timeout_seconds, cpu_threads, unload_after_task, eval_mode,
        )
        run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        total = len(model_specs) * len(selected_cases)
        completed = 0
        errors = 0
        started_at = time.monotonic()

        # Models are processed serially.  Do not turn this into a worker pool:
        # loading two vision models concurrently is the main source of OOMs on
        # the supported CPU-only developer machines.
        for model_spec in model_specs:
            model = None
            model_name = str(model_spec).split(":", 1)[-1]
            try:
                model = self.registry.create(
                    model_spec,
                    mock_noise=mock_noise,
                    model_prompt=model_prompt,
                    cpu_threads=cpu_threads,
                    unload_after_task=unload_after_task,
                    # Forward the same budget to providers that support a
                    # native network timeout (notably Ollama). The runner
                    # timeout alone cannot cancel an HTTP request safely.
                    timeout_seconds=timeout_seconds,
                )
                model_name = model.model_name
            except Exception as exc:
                LOGGER.exception("Model initialization failed | spec=%s", model_spec)
                for case in selected_cases:
                    inference = InferenceResult(
                        text="", latency_seconds=0.0,
                        status=InferenceStatus.FAILED,
                        error=f"Model initialization failed: {type(exc).__name__}: {exc}",
                        device="unknown",
                    )
                    result = self._evaluate(run_id, model_name, case, inference, eval_mode)
                    completed += 1
                    errors += 1
                    yield RunnerProgress(
                        run_id=run_id, completed=completed, total=total,
                        model_name=model_name, case=case, result=result.to_dict(),
                        stage="completed", elapsed_seconds=time.monotonic() - started_at,
                        estimated_remaining_seconds=None, error_count=errors,
                    )
                    if max_errors is not None and max_errors > 0 and errors >= max_errors:
                        return
                continue

            LOGGER.info("Model started | model=%s | cases=%d", model.model_name, len(selected_cases))
            try:
                for case in selected_cases:
                    LOGGER.debug(
                        "Inference started | model=%s | image=%s | completed=%d/%d",
                        model.model_name, case.image_path, completed, total,
                    )
                    if progress:
                        progress(completed, total, f"{model.model_name}: {case.image_path}")
                    elapsed_before = time.monotonic() - started_at
                    yield RunnerProgress(
                        run_id=run_id,
                        completed=completed,
                        total=total,
                        model_name=model.model_name,
                        case=case,
                        result=None,
                        stage="processing",
                        elapsed_seconds=elapsed_before,
                        estimated_remaining_seconds=(
                            (elapsed_before / completed) * (total - completed)
                            if completed
                            else None
                        ),
                        error_count=errors,
                    )
                    try:
                        # The adapter boundary is deliberately narrow.  Any
                        # provider exception becomes a failed document rather
                        # than stopping the whole run.
                        raw = self._perform_with_timeout(
                            model,
                            case.image_path,
                            timeout_seconds,
                            late_result=lambda late_raw, late_error: self._emit_trace(
                                trace,
                                run_id,
                                model.model_name,
                                case,
                                late_raw,
                                timing="late_after_timeout",
                                error=late_error,
                            ),
                        )
                        inference = (
                            raw
                            if isinstance(raw, InferenceResult)
                            else InferenceResult.from_legacy_dict(raw)
                        )
                    except Exception as exc:  # Adapter boundary: keep one failure local.
                        LOGGER.exception(
                            "Adapter exception | model=%s | image=%s",
                            model.model_name, case.image_path,
                        )
                        inference = InferenceResult(
                            text="",
                            latency_seconds=0.0,
                            status=InferenceStatus.FAILED,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    self._emit_trace(
                        trace,
                        run_id,
                        model.model_name,
                        case,
                        inference,
                        timing="on_time",
                    )
                    result = self._evaluate(
                        run_id, model.model_name, case, inference, eval_mode
                    )
                    completed += 1
                    result_dict = result.to_dict()
                    if inference.status is not InferenceStatus.SUCCESS:
                        errors += 1
                        LOGGER.warning(
                            "Inference finished with non-success status | model=%s | image=%s | status=%s | error=%s",
                            model.model_name, case.image_path, inference.status.value, inference.error,
                        )
                    else:
                        LOGGER.info(
                            "Inference completed | model=%s | image=%s | latency=%.3fs | chars=%d | tokens=%s",
                            model.model_name, case.image_path, inference.latency_seconds,
                            len(inference.text or ""), inference.output_tokens,
                        )
                    elapsed = time.monotonic() - started_at
                    remaining = (
                        (elapsed / completed) * (total - completed)
                        if completed and completed < total
                        else 0.0
                    )
                    yield RunnerProgress(
                        run_id=run_id,
                        completed=completed,
                        total=total,
                        model_name=model.model_name,
                        case=case,
                        result=result_dict,
                        stage="completed",
                        elapsed_seconds=elapsed,
                        estimated_remaining_seconds=remaining,
                        error_count=errors,
                    )
                    if max_errors is not None and max_errors > 0 and errors >= max_errors:
                        return
            finally:
                close = getattr(model, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        LOGGER.exception("Model cleanup failed | model=%s", model.model_name)
                LOGGER.info("Model released | model=%s", model.model_name)

    @staticmethod
    def _perform_with_timeout(
        model,
        image_path: str,
        timeout_seconds: float | None,
        late_result: Callable[[Any | None, str | None], None] | None = None,
    ):
        if timeout_seconds is None or timeout_seconds <= 0:
            return model.perform_ocr(image_path)

        result_holder: list[Any] = []
        error_holder: list[BaseException] = []
        finished = threading.Event()

        def worker() -> None:
            try:
                result_holder.append(model.perform_ocr(image_path))
            except BaseException as exc:  # propagate through the caller thread
                error_holder.append(exc)
            finally:
                finished.set()

        # A Python thread cannot safely be force-killed.  The daemon flag lets
        # the UI continue after a timeout; provider-specific timeouts (Ollama,
        # HTTP clients, etc.) must still be configured to stop the real call.
        thread = threading.Thread(target=worker, name="ocr-inference", daemon=True)
        thread.start()
        if finished.wait(timeout_seconds):
            if error_holder:
                raise error_holder[0]
            return result_holder[0]

        if not finished.is_set():
            LOGGER.warning("Inference timeout | model=%s | image=%s | timeout=%.2fs", getattr(model, "model_name", "unknown"), image_path, timeout_seconds)
            if late_result:
                # Keep observing the provider in the background only to persist
                # its eventual raw response. It is never reintroduced into the
                # quality score or the progress counter.
                def capture_late() -> None:
                    finished.wait()
                    try:
                        if error_holder:
                            raise error_holder[0]
                        late_result(result_holder[0] if result_holder else None, None)
                    except Exception as exc:
                        late_result(None, f"{type(exc).__name__}: {exc}")

                threading.Thread(target=capture_late, name="ocr-late-result", daemon=True).start()
            return InferenceResult(
                    text="",
                    latency_seconds=timeout_seconds,
                    status=InferenceStatus.TIMEOUT,
                    error=f"Timeout after {timeout_seconds:.1f} seconds; late output is persisted when available",
                    device=getattr(model, "device_name", "unknown"),
                )

    @staticmethod
    def _emit_trace(
        trace: TraceCallback | None,
        run_id: str,
        model_name: str,
        case: BenchmarkCase,
        raw: Any | None,
        *,
        timing: str,
        error: str | None = None,
    ) -> None:
        if trace is None:
            return
        try:
            inference = (
                raw
                if isinstance(raw, InferenceResult)
                else InferenceResult.from_legacy_dict(raw)
                if isinstance(raw, dict)
                else None
            )
            trace(
                {
                    "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "run_id": run_id,
                    "model": model_name,
                    "image_path": case.image_path,
                    "category": case.category,
                    "timing": timing,
                    "status": inference.status.value if inference else "failed",
                    "latency": inference.latency_seconds if inference else None,
                    "text": inference.text if inference else "",
                    "reasoning": inference.reasoning if inference else None,
                    "raw_response": inference.raw_response if inference else None,
                    "input_tokens": inference.input_tokens if inference else None,
                    "output_tokens": inference.output_tokens if inference else None,
                    "tokens_per_second": (
                        inference.tokens_per_second if inference else None
                    ),
                    "error": error or (inference.error if inference else None),
                }
            )
        except Exception:
            # Trace persistence must never break the benchmark itself.
            return

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
            raw_response=inference.raw_response,
            reasoning=inference.reasoning,
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
