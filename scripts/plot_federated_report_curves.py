from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Series:
    label: str
    x: list[float]
    y: list[float]
    color: str


ATTACK_COLORS = ["#d04a02", "#f28e2b", "#e15759", "#b55d60", "#ff9d76"]
DEFENSE_COLORS = ["#4e79a7", "#59a14f", "#76b7b2", "#2f6b8a", "#3d8c40"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render SVG training curves from a federated benchmark JSON report.")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _extract_xy(traces: Iterable[dict], x_key: str, y_key: str) -> tuple[list[float], list[float]]:
    x_values: list[float] = []
    y_values: list[float] = []
    for trace in traces:
        if x_key in trace and y_key in trace:
            x_values.append(float(trace[x_key]))
            y_values.append(float(trace[y_key]))
    return x_values, y_values


def _budget_label(prefix: str, budget: object) -> str:
    try:
        budget_value = float(budget)
    except (TypeError, ValueError):
        return prefix
    return f"{prefix} {budget_value:g}"


def collect_metric_series(report: dict, metric_name: str) -> list[Series]:
    series: list[Series] = []

    clean = report.get("clean")
    if isinstance(clean, dict):
        traces_key = "training_traces" if metric_name == "mean_loss" else "round_traces"
        x_values, y_values = _extract_xy(clean.get(traces_key, []), "round_index", metric_name)
        if x_values:
            series.append(Series(label="clean", x=x_values, y=y_values, color="#222222"))

    for index, attack in enumerate(report.get("attacks", [])):
        if not isinstance(attack, dict):
            continue
        traces_key = "training_traces" if metric_name == "mean_loss" else "round_traces"
        x_values, y_values = _extract_xy(attack.get(traces_key, []), "round_index", metric_name)
        if x_values:
            series.append(
                Series(
                    label=_budget_label("attack", attack.get("attack_budget")),
                    x=x_values,
                    y=y_values,
                    color=ATTACK_COLORS[index % len(ATTACK_COLORS)],
                )
            )

    for index, defended in enumerate(report.get("defended_attacks", [])):
        if not isinstance(defended, dict):
            continue
        traces_key = "training_traces" if metric_name == "mean_loss" else "round_traces"
        x_values, y_values = _extract_xy(defended.get(traces_key, []), "round_index", metric_name)
        if x_values:
            defense_name = defended.get("defense_method", "defense")
            label = f"{defense_name} {_budget_label('', defended.get('attack_budget')).strip()}".strip()
            series.append(
                Series(
                    label=label,
                    x=x_values,
                    y=y_values,
                    color=DEFENSE_COLORS[index % len(DEFENSE_COLORS)],
                )
            )

    return series


def _nice_bounds(values: list[float]) -> tuple[float, float]:
    minimum = min(values)
    maximum = max(values)
    if minimum == maximum:
        padding = 1.0 if minimum == 0.0 else abs(minimum) * 0.1
        return minimum - padding, maximum + padding
    padding = (maximum - minimum) * 0.08
    return minimum - padding, maximum + padding


def render_svg_line_chart(
    *,
    title: str,
    x_label: str,
    y_label: str,
    series: list[Series],
    output_path: Path,
) -> None:
    if not series:
        return

    width = 1120
    height = 720
    margin_left = 86
    margin_right = 280
    margin_top = 64
    margin_bottom = 72
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    all_x = [value for trace in series for value in trace.x]
    all_y = [value for trace in series for value in trace.y]
    min_x, max_x = _nice_bounds(all_x)
    min_y, max_y = _nice_bounds(all_y)

    def x_to_px(value: float) -> float:
        return margin_left + (value - min_x) / max(max_x - min_x, 1e-9) * plot_width

    def y_to_px(value: float) -> float:
        return margin_top + plot_height - (value - min_y) / max(max_y - min_y, 1e-9) * plot_height

    x_ticks = sorted(set(all_x))
    if len(x_ticks) > 8:
        step = max(len(x_ticks) // 8, 1)
        x_ticks = x_ticks[::step]
        if x_ticks[-1] != all_x[-1]:
            x_ticks.append(sorted(set(all_x))[-1])

    y_ticks = [min_y + (max_y - min_y) * fraction / 4.0 for fraction in range(5)]

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{margin_left}" y="34" font-size="24" font-family="Arial, sans-serif" font-weight="bold">{title}</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#333" stroke-width="2"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#333" stroke-width="2"/>',
    ]

    for tick in y_ticks:
        y_px = y_to_px(tick)
        svg_lines.append(
            f'<line x1="{margin_left}" y1="{y_px:.2f}" x2="{margin_left + plot_width}" y2="{y_px:.2f}" stroke="#dddddd" stroke-width="1"/>'
        )
        svg_lines.append(
            f'<text x="{margin_left - 10}" y="{y_px + 4:.2f}" text-anchor="end" font-size="12" font-family="Arial, sans-serif">{tick:.4g}</text>'
        )

    for tick in x_ticks:
        x_px = x_to_px(tick)
        svg_lines.append(
            f'<line x1="{x_px:.2f}" y1="{margin_top}" x2="{x_px:.2f}" y2="{margin_top + plot_height}" stroke="#f0f0f0" stroke-width="1"/>'
        )
        svg_lines.append(
            f'<text x="{x_px:.2f}" y="{margin_top + plot_height + 22}" text-anchor="middle" font-size="12" font-family="Arial, sans-serif">{tick:g}</text>'
        )

    for trace in series:
        points = " ".join(f"{x_to_px(x):.2f},{y_to_px(y):.2f}" for x, y in zip(trace.x, trace.y))
        svg_lines.append(
            f'<polyline fill="none" stroke="{trace.color}" stroke-width="3" points="{points}"/>'
        )
        for x_value, y_value in zip(trace.x, trace.y):
            svg_lines.append(
                f'<circle cx="{x_to_px(x_value):.2f}" cy="{y_to_px(y_value):.2f}" r="3.5" fill="{trace.color}"/>'
            )

    svg_lines.append(
        f'<text x="{margin_left + plot_width / 2:.2f}" y="{height - 18}" text-anchor="middle" font-size="15" font-family="Arial, sans-serif">{x_label}</text>'
    )
    svg_lines.append(
        f'<text x="24" y="{margin_top + plot_height / 2:.2f}" transform="rotate(-90 24 {margin_top + plot_height / 2:.2f})" text-anchor="middle" font-size="15" font-family="Arial, sans-serif">{y_label}</text>'
    )

    legend_x = margin_left + plot_width + 24
    legend_y = margin_top + 16
    for index, trace in enumerate(series):
        y_offset = legend_y + index * 24
        svg_lines.append(
            f'<line x1="{legend_x}" y1="{y_offset}" x2="{legend_x + 18}" y2="{y_offset}" stroke="{trace.color}" stroke-width="3"/>'
        )
        svg_lines.append(
            f'<text x="{legend_x + 26}" y="{y_offset + 4}" font-size="13" font-family="Arial, sans-serif">{trace.label}</text>'
        )

    svg_lines.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(svg_lines), encoding="utf-8")
    print(f"wrote {output_path}")


def main() -> None:
    args = parse_args()
    report = _read_json(args.report)
    output_dir = args.output_dir or args.report.with_suffix("")
    output_dir.mkdir(parents=True, exist_ok=True)

    charts = [
        ("mean_loss", "Training Loss by Round", "Round", "Mean BPR loss", "training_mean_loss.svg"),
        ("overall_hr@k", "Overall Benign HR by Round", "Round", "HR@K", "overall_hr_by_round.svg"),
        ("target_hitrate@10", "Target Hit@10 by Round", "Round", "Target Hit@10", "target_hit10_by_round.svg"),
        ("target_mean_rank", "Target Mean Rank by Round", "Round", "Target mean rank", "target_mean_rank_by_round.svg"),
    ]

    for metric_name, title, x_label, y_label, filename in charts:
        series = collect_metric_series(report, metric_name)
        render_svg_line_chart(
            title=title,
            x_label=x_label,
            y_label=y_label,
            series=series,
            output_path=output_dir / filename,
        )


if __name__ == "__main__":
    main()
