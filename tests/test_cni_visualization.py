from __future__ import annotations

from ocr_benchmark.visualization import (
    cni_accuracy_chart,
    cni_error_rate_chart,
    cni_field_accuracy_chart,
    cni_latency_chart,
    cni_quality_latency_chart,
    cni_reliability_chart,
)


def _results() -> list[dict]:
    return [
        {
            "model": "model-a",
            "status": "success",
            "accuracy": 1.0,
            "text_similarity": 0.98,
            "cer": 0.02,
            "wer": 0.1,
            "end_to_end_seconds": 2.5,
            "field_comparison": {
                "rows": [
                    {"field": "cin", "state": "correct"},
                    {"field": "nom", "state": "correct"},
                ]
            },
        },
        {
            "model": "model-b",
            "status": "invalid_json",
            "accuracy": 0.5,
            "text_similarity": 0.75,
            "cer": 0.25,
            "wer": 0.5,
            "end_to_end_seconds": 1.2,
            "field_comparison": {
                "rows": [
                    {"field": "cin", "state": "correct"},
                    {"field": "nom", "state": "different"},
                ]
            },
        },
    ]


def test_cni_overview_charts_render_available_metrics():
    results = _results()
    chart_functions = (
        cni_accuracy_chart,
        cni_error_rate_chart,
        cni_field_accuracy_chart,
        cni_latency_chart,
        cni_quality_latency_chart,
        cni_reliability_chart,
    )

    for chart_function in chart_functions:
        figure = chart_function(results)
        assert figure.data, chart_function.__name__


def test_cni_charts_accept_the_same_filtered_subset():
    filtered = [_results()[0]]

    quality = cni_accuracy_chart(filtered)
    latency = cni_latency_chart(filtered)
    reliability = cni_reliability_chart(filtered)

    assert {value for trace in quality.data for value in trace.x} == {"model-a"}
    assert {value for trace in latency.data for value in trace.x} == {"model-a"}
    assert {value for trace in reliability.data for value in trace.x} == {"model-a"}
