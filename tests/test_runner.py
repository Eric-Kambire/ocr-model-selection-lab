import threading
import time

from ocr_benchmark.domain import BenchmarkCase
from ocr_benchmark.registry import ModelRegistry
from ocr_benchmark.runner import BenchmarkRunner, summarize_results


class SuccessfulModel:
    model_name = "successful"

    def perform_ocr(self, image_path):
        return {
            "text": "expected text",
            "latency": 0.5,
            "status": "success",
            "device": "cpu",
        }


class FailingModel:
    model_name = "failing"

    def perform_ocr(self, image_path):
        return {
            "text": "",
            "latency": 0.1,
            "status": "failed",
            "error": "provider unavailable",
            "device": "cpu",
        }


class SlowThinkingModel:
    model_name = "slow-thinking"

    def perform_ocr(self, image_path):
        time.sleep(0.08)
        return {
            "text": "late transcription",
            "thinking": "internal model trace",
            "raw_response": "complete raw provider response",
            "latency": 0.08,
            "status": "success",
            "device": "cpu",
        }


def test_mock_path_normalization_accepts_windows_and_linux_separators():
    from models.mock_model import MockOCRModel

    assert MockOCRModel._normalize_path(r"dataset\tables\image.png") == (
        MockOCRModel._normalize_path("dataset/tables/image.png")
    )


def _runner():
    registry = ModelRegistry()
    registry.register("ok", lambda model_name, **options: SuccessfulModel())
    registry.register("fail", lambda model_name, **options: FailingModel())
    return BenchmarkRunner(registry)


def test_runner_does_not_score_technical_failures():
    case = BenchmarkCase("image.png", "expected text", "test")
    _, results = _runner().run(["ok:model", "fail:model"], [case])
    assert results[0]["accuracy"] == 1
    assert results[1]["accuracy"] is None
    assert results[1]["status"] == "failed"


def test_summary_separates_quality_and_reliability():
    case = BenchmarkCase("image.png", "expected text", "test")
    _, results = _runner().run(["ok:model", "fail:model"], [case])
    summary = summarize_results(results).set_index("Model")
    assert summary.loc["successful", "Success rate"] == 1
    assert summary.loc["failing", "Success rate"] == 0


def test_stream_reports_processing_before_completion():
    case = BenchmarkCase("image.png", "expected text", "test")
    updates = list(_runner().iter_run(["ok:model"], [case]))
    assert [update.stage for update in updates] == ["processing", "completed"]
    assert updates[0].result is None
    assert updates[1].result["accuracy"] == 1


def test_timeout_keeps_late_output_in_trace():
    registry = ModelRegistry()
    registry.register("slow", lambda model_name, **options: SlowThinkingModel())
    runner = BenchmarkRunner(registry)
    traces = []
    late_received = threading.Event()

    def capture(event):
        traces.append(event)
        if event["timing"] == "late_after_timeout":
            late_received.set()

    case = BenchmarkCase("image.png", "late transcription", "test")
    _, results = runner.run(
        ["slow:model"],
        [case],
        timeout_seconds=0.01,
        trace=capture,
    )

    assert results[0]["status"] == "timeout"
    assert late_received.wait(1)
    late = next(item for item in traces if item["timing"] == "late_after_timeout")
    assert late["text"] == "late transcription"
    assert late["reasoning"] == "internal model trace"
    assert late["raw_response"] == "complete raw provider response"
