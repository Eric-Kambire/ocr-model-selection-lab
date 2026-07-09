from __future__ import annotations

import math

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
