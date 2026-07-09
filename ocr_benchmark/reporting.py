from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


def save_run(
    run_id: str,
    summary: pd.DataFrame,
    results: list[dict[str, Any]],
    output_root: str | Path = "runs",
) -> Path:
    """Persist a run atomically in its own directory."""
    run_dir = Path(output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _atomic_text(run_dir / "results.json", json.dumps(results, indent=2, ensure_ascii=False))
    summary.to_csv(run_dir / "summary.csv", index=False)
    pd.DataFrame(results).to_csv(run_dir / "details.csv", index=False)
    _atomic_text(run_dir / "report.md", render_markdown(run_id, summary, results))
    return run_dir


def render_markdown(
    run_id: str,
    summary: pd.DataFrame,
    results: list[dict[str, Any]],
) -> str:
    lines = [
        f"# OCR benchmark run `{run_id}`",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Metric definitions",
        "",
        "- **Quality score**: `1 - CER` in Standard mode, or the weighted applicable banking metrics in Bankmark mode.",
        "- **CER**: character edits divided by reference characters. Lower is better and values can exceed 100% when a model hallucinates.",
        "- **WER**: word edits divided by reference words. Lower is better.",
        "- **Latency**: wall-clock seconds required for one document. Lower is better.",
        "- **P95 latency**: 95% of successful documents completed in this time or less.",
        "- **Documents/s**: successful documents processed per second, measured sequentially.",
        "- **Tokens/s**: generated language-model tokens per second when the provider exposes token counters. It is not applicable to classic OCR engines.",
        "- **Success rate**: successful technical executions divided by attempted executions.",
        "",
        "## Failures",
        "",
    ]
    failures = [result for result in results if result["status"] != "success"]
    if not failures:
        lines.append("No technical failures.")
    else:
        for result in failures:
            lines.append(
                f"- `{result['model']}` / `{Path(result['image_path']).name}`: "
                f"{result.get('error') or result['status']}"
            )
    lines.append("")
    return "\n".join(lines)


def _atomic_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)
