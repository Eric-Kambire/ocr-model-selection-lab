from __future__ import annotations

import math
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

COLORS = ["#6366F1", "#14B8A6", "#F59E0B", "#EC4899", "#3B82F6", "#8B5CF6"]


def empty_figure(message: str = "Run a benchmark to display this chart.") -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(text=message, x=0.5, y=0.5, showarrow=False, font={"size": 16})
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    return style_figure(figure)


def quality_speed_chart(summary: pd.DataFrame) -> go.Figure:
    if summary.empty:
        return empty_figure()
    plot = summary.dropna(subset=["Quality score", "Documents/s"]).copy()
    if plot.empty:
        return empty_figure("No successful run has both quality and speed data.")
    plot["Quality (%)"] = plot["Quality score"] * 100
    figure = px.scatter(
        plot,
        x="Documents/s",
        y="Quality (%)",
        color="Model",
        size="Success rate",
        hover_data=["Median latency (s)", "P95 latency (s)", "Device"],
        title="Quality versus processing speed",
        color_discrete_sequence=COLORS,
    )
    figure.update_traces(marker={"line": {"width": 1, "color": "white"}})
    return style_figure(figure)


def latency_chart(summary: pd.DataFrame) -> go.Figure:
    if summary.empty:
        return empty_figure()
    figure = go.Figure()
    for index, row in summary.iterrows():
        figure.add_trace(
            go.Bar(
                name=str(row["Model"]),
                x=["Mean", "Median", "P95"],
                y=[
                    row["Mean latency (s)"],
                    row["Median latency (s)"],
                    row["P95 latency (s)"],
                ],
                marker_color=COLORS[index % len(COLORS)],
                hovertemplate="%{x}: %{y:.3f}s<extra>%{fullData.name}</extra>",
            )
        )
    figure.update_layout(title="Latency distribution summary", barmode="group", yaxis_title="Seconds")
    return style_figure(figure)


def reliability_chart(summary: pd.DataFrame) -> go.Figure:
    if summary.empty:
        return empty_figure()
    plot = summary.copy()
    plot["Successful (%)"] = plot["Success rate"] * 100
    plot["Failed (%)"] = 100 - plot["Successful (%)"]
    figure = go.Figure(
        [
            go.Bar(
                name="Successful",
                x=plot["Model"],
                y=plot["Successful (%)"],
                marker_color="#14B8A6",
            ),
            go.Bar(
                name="Failed",
                x=plot["Model"],
                y=plot["Failed (%)"],
                marker_color="#EF4444",
            ),
        ]
    )
    figure.update_layout(title="Technical reliability", barmode="stack", yaxis_title="Executions (%)")
    return style_figure(figure)


def category_quality_chart(results: list[dict]) -> go.Figure:
    if not results:
        return empty_figure()
    frame = pd.DataFrame(results)
    frame = frame[(frame["status"] == "success") & frame["accuracy"].notna()]
    if frame.empty:
        return empty_figure("No successful result is available.")
    grouped = frame.groupby(["model", "category"], as_index=False)["accuracy"].mean()
    grouped["Quality (%)"] = grouped["accuracy"] * 100
    figure = px.bar(
        grouped,
        x="category",
        y="Quality (%)",
        color="model",
        barmode="group",
        title="Quality by document category",
        color_discrete_sequence=COLORS,
    )
    return style_figure(figure)


def cni_accuracy_chart(results: list[dict]) -> go.Figure:
    """Compare l'exactitude stricte et la similarité textuelle par modèle."""
    if not results:
        return empty_figure("Lancez un benchmark CNI pour afficher les résultats.")
    frame = pd.DataFrame(results)
    available = [
        metric
        for metric in ("accuracy", "text_similarity")
        if metric in frame and frame[metric].notna().any()
    ]
    if not available:
        return empty_figure("Aucun score : les résultats filtrés ne sont pas notés.")
    grouped = frame.groupby("model", as_index=False)[available].mean()
    labels = {
        "accuracy": "Exactitude des champs",
        "text_similarity": "Similarité textuelle",
    }
    plot = grouped.melt(
        id_vars="model",
        value_vars=available,
        var_name="metric",
        value_name="value",
    )
    plot["Mesure"] = plot["metric"].map(labels)
    plot["Score (%)"] = plot["value"] * 100
    figure = px.bar(
        plot,
        x="model",
        y="Score (%)",
        color="Mesure",
        barmode="group",
        title="Qualité moyenne par modèle",
        labels={"model": "Modèle"},
        color_discrete_sequence=COLORS,
    )
    return style_figure(figure)


def cni_latency_chart(results: list[dict]) -> go.Figure:
    """Montre la distribution réelle du temps bout-en-bout par modèle."""
    if not results:
        return empty_figure("Lancez un benchmark CNI pour afficher les latences.")
    frame = pd.DataFrame(results)
    latency_column = (
        "end_to_end_seconds"
        if "end_to_end_seconds" in frame
        and frame["end_to_end_seconds"].notna().any()
        else "latency"
    )
    if latency_column not in frame or frame[latency_column].dropna().empty:
        return empty_figure("Aucune latence CNI disponible.")
    plot = frame.dropna(subset=[latency_column]).copy()
    figure = px.box(
        plot,
        x="model",
        y=latency_column,
        color="model",
        points="all",
        title="Distribution du temps bout-en-bout",
        labels={latency_column: "Secondes", "model": "Modèle"},
        color_discrete_sequence=COLORS,
    )
    return style_figure(figure)


def cni_error_rate_chart(results: list[dict]) -> go.Figure:
    """Affiche CER et WER ; plus bas signifie une meilleure transcription."""
    if not results:
        return empty_figure("Aucun résultat CNI à afficher.")
    frame = pd.DataFrame(results)
    available = [
        metric
        for metric in ("cer", "wer")
        if metric in frame and frame[metric].notna().any()
    ]
    if not available:
        return empty_figure("CER/WER indisponibles pour les résultats filtrés.")
    grouped = frame.groupby("model", as_index=False)[available].mean()
    plot = grouped.melt(
        id_vars="model",
        value_vars=available,
        var_name="Mesure",
        value_name="value",
    )
    plot["Mesure"] = plot["Mesure"].str.upper()
    plot["Taux d'erreur (%)"] = plot["value"] * 100
    figure = px.bar(
        plot,
        x="model",
        y="Taux d'erreur (%)",
        color="Mesure",
        barmode="group",
        title="Erreurs de transcription",
        labels={"model": "Modèle"},
        color_discrete_sequence=["#F59E0B", "#EF4444"],
    )
    return style_figure(figure)


def cni_field_accuracy_chart(results: list[dict]) -> go.Figure:
    """Heatmap des taux de champs exacts, après application des filtres."""
    records: list[dict[str, Any]] = []
    for result in results or []:
        comparison = result.get("field_comparison")
        rows = comparison.get("rows", []) if isinstance(comparison, dict) else []
        for row in rows:
            if not isinstance(row, dict) or row.get("state") == "reference_missing":
                continue
            records.append(
                {
                    "model": result.get("model", "—"),
                    "field": row.get("field", "—"),
                    "correct": 1.0 if row.get("state") == "correct" else 0.0,
                }
            )
    if not records:
        return empty_figure("Aucun champ comparable dans les résultats filtrés.")
    frame = pd.DataFrame(records)
    pivot = (
        frame.groupby(["model", "field"])["correct"]
        .mean()
        .mul(100)
        .unstack(fill_value=float("nan"))
    )
    figure = go.Figure(
        go.Heatmap(
            z=pivot.to_numpy(),
            x=list(pivot.columns),
            y=list(pivot.index),
            zmin=0,
            zmax=100,
            colorscale=[
                [0.0, "#FEE2E2"],
                [0.5, "#FEF3C7"],
                [1.0, "#D1FAE5"],
            ],
            colorbar={"title": "Correct (%)"},
            hovertemplate=(
                "Modèle=%{y}<br>Champ=%{x}<br>Correct=%{z:.1f}%<extra></extra>"
            ),
        )
    )
    figure.update_layout(title="Exactitude par champ")
    return style_figure(figure)


def cni_quality_latency_chart(results: list[dict]) -> go.Figure:
    """Positionne les modèles selon qualité et temps pour aider la sélection."""
    if not results:
        return empty_figure("Aucun résultat CNI à afficher.")
    frame = pd.DataFrame(results)
    latency_column = (
        "end_to_end_seconds"
        if "end_to_end_seconds" in frame
        and frame["end_to_end_seconds"].notna().any()
        else "latency"
    )
    if (
        "accuracy" not in frame
        or latency_column not in frame
        or frame.dropna(subset=["accuracy", latency_column]).empty
    ):
        return empty_figure("Qualité et temps doivent être disponibles ensemble.")
    grouped = (
        frame.dropna(subset=["accuracy", latency_column])
        .groupby("model", as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            latency=(latency_column, "mean"),
            samples=("accuracy", "size"),
        )
    )
    grouped["Exactitude (%)"] = grouped["accuracy"] * 100
    figure = px.scatter(
        grouped,
        x="latency",
        y="Exactitude (%)",
        color="model",
        size="samples",
        text="model",
        title="Compromis qualité / temps",
        labels={
            "latency": "Temps moyen bout-en-bout (s)",
            "model": "Modèle",
            "samples": "Cas notés",
        },
        color_discrete_sequence=COLORS,
    )
    figure.update_traces(textposition="top center")
    return style_figure(figure)


def cni_reliability_chart(results: list[dict]) -> go.Figure:
    """Répartit les statuts techniques par modèle en pourcentage."""
    if not results:
        return empty_figure("Aucun résultat CNI à afficher.")
    frame = pd.DataFrame(results)
    if "status" not in frame or "model" not in frame:
        return empty_figure("Statuts techniques indisponibles.")
    counts = (
        frame.groupby(["model", "status"])
        .size()
        .rename("count")
        .reset_index()
    )
    totals = counts.groupby("model")["count"].transform("sum")
    counts["Exécutions (%)"] = counts["count"] / totals * 100
    figure = px.bar(
        counts,
        x="model",
        y="Exécutions (%)",
        color="status",
        barmode="stack",
        title="Fiabilité technique",
        labels={"model": "Modèle", "status": "Statut"},
        color_discrete_map={
            "success": "#10B981",
            "invalid_json": "#F59E0B",
            "timeout": "#F97316",
            "failed": "#EF4444",
        },
    )
    return style_figure(figure)


def style_figure(figure: go.Figure) -> go.Figure:
    figure.update_layout(
        template="plotly_white",
        font={"family": "Inter, system-ui, sans-serif", "color": "#1F2937"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,250,252,0.75)",
        margin={"l": 50, "r": 25, "t": 65, "b": 50},
        legend_title_text="",
        hoverlabel={"bgcolor": "white", "font_size": 13},
    )
    figure.update_xaxes(gridcolor="#E5E7EB")
    figure.update_yaxes(gridcolor="#E5E7EB")
    return figure
