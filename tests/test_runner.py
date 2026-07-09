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
