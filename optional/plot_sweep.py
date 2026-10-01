"""Render detection-curve small multiples from two `qc sweep` reports.

Writes a light and a dark SVG (GitHub serves the right one through
<picture>). Pure Python, no plotting dependency. The sweep JSON is the table
view; the SVG is a picture of it.

    python -m optional.plot_sweep reports/sweep/small.json \
        reports/sweep/realistic.json --out docs/img/detection-curves
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

PANELS = (
    ("market_movement", "Market movement", "drop of one commodity"),
    ("coding_error", "Coding error", "dollar cut on affected products"),
    ("warehouse_transform_error", "Warehouse transform", "dollar cut on one commodity"),
    ("week_restatement", "One week restated", "cut of one overlap week"),
)
THEMES = {
    "light": {
        "surface": "#fcfcfb", "primary": "#0b0b0b", "secondary": "#52514e",
        "muted": "#8a8984", "grid": "#e6e5e1", "series": ("#2a78d6", "#eb6834"),
    },
    "dark": {
        "surface": "#1a1a19", "primary": "#ffffff", "secondary": "#c3c2b7",
        "muted": "#8f8e86", "grid": "#33332f", "series": ("#3987e5", "#d95926"),
    },
}
PANEL_W, PANEL_H = 300, 190
PLOT_LEFT, PLOT_RIGHT, PLOT_TOP, PLOT_BOTTOM = 44, 70, 44, 40
GAP_X, GAP_Y = 24, 18
HEADER, FOOTER, MARGIN = 74, 44, 20
X_MIN, X_MAX = 0.01, 0.4


def _x(magnitude: float) -> float:
    span = PANEL_W - PLOT_LEFT - PLOT_RIGHT
    return PLOT_LEFT + span * (math.log(magnitude) - math.log(X_MIN)) / (
        math.log(X_MAX) - math.log(X_MIN)
    )


def _y(rate: float) -> float:
    return PLOT_TOP + (PANEL_H - PLOT_TOP - PLOT_BOTTOM) * (1.0 - rate)


def _text(x: float, y: float, value: str, fill: str, size: int = 11,
          anchor: str = "start", weight: int = 400) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}">{value}</text>'
    )


def _panel(report_pair: list[dict[str, Any]], family: str, title: str,
           subtitle: str, theme: dict[str, Any], labels: tuple[str, str]) -> str:
    parts = [
        _text(0, 14, title, theme["primary"], 13, weight=600),
        _text(0, 29, f"x: {subtitle}", theme["secondary"], 10),
    ]
    for rate in (0.0, 0.5, 1.0):
        y = _y(rate)
        parts.append(
            f'<line x1="{PLOT_LEFT}" x2="{PANEL_W - PLOT_RIGHT}" y1="{y:.1f}" '
            f'y2="{y:.1f}" stroke="{theme["grid"]}" stroke-width="1"/>'
        )
        parts.append(_text(PLOT_LEFT - 6, y + 4, f"{rate:.0%}", theme["secondary"], 10, "end"))
    for tick in (0.01, 0.02, 0.05, 0.1, 0.2, 0.4):
        parts.append(
            _text(_x(tick), PANEL_H - PLOT_BOTTOM + 15, f"{tick:.0%}",
                  theme["secondary"], 10, "middle")
        )
    ends: list[tuple[float, str, str]] = []
    for index, report in enumerate(report_pair):
        color = theme["series"][index]
        points = report["summary"]["curves"].get(family, [])
        if not points:
            continue
        upper = [(_x(p["magnitude"]), _y(p["detection"]["ci_high"])) for p in points]
        lower = [(_x(p["magnitude"]), _y(p["detection"]["ci_low"])) for p in reversed(points)]
        band = " ".join(f"{x:.1f},{y:.1f}" for x, y in upper + lower)
        parts.append(f'<polygon points="{band}" fill="{color}" fill-opacity="0.10"/>')
        line = " ".join(
            f"{_x(p['magnitude']):.1f},{_y(p['detection']['rate']):.1f}" for p in points
        )
        parts.append(
            f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="2" '
            f'stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for point in points:
            parts.append(
                f'<circle cx="{_x(point["magnitude"]):.1f}" '
                f'cy="{_y(point["detection"]["rate"]):.1f}" r="4" fill="{color}" '
                f'stroke="{theme["surface"]}" stroke-width="2"/>'
            )
        last = points[-1]
        ends.append((_y(last["detection"]["rate"]), labels[index], color))
    # Direct end labels only where the two series end apart (otherwise the
    # legend carries identity), nudged so they never collide.
    if len(ends) == 2 and abs(ends[0][0] - ends[1][0]) >= 12:
        for y, label, color in ends:
            x = PANEL_W - PLOT_RIGHT + 8
            parts.append(
                f'<line x1="{x}" x2="{x + 10}" y1="{y:.1f}" y2="{y:.1f}" '
                f'stroke="{color}" stroke-width="2"/>'
            )
            parts.append(_text(x + 14, y + 4, label, theme["secondary"], 10))
    return "".join(parts)


def render(reports: list[dict[str, Any]], mode: str) -> str:
    theme = THEMES[mode]
    labels = (reports[0]["profile"], reports[1]["profile"])
    width = MARGIN * 2 + PANEL_W * 2 + GAP_X
    height = HEADER + PANEL_H * 2 + GAP_Y + FOOTER
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Inter, -apple-system, '
        f'Segoe UI, Helvetica, Arial, sans-serif" role="img" '
        f'aria-label="Detection rate against fault size, by family and profile">',
        f'<rect width="{width}" height="{height}" fill="{theme["surface"]}"/>',
        _text(MARGIN, 26, "Detection rate vs. fault size", theme["primary"], 16, weight=600),
        _text(MARGIN, 44, "Final engine status on held-out seeds, 10 per point; shaded: 95% Wilson interval",
              theme["secondary"], 11),
    ]
    legend_x = MARGIN
    for index, label in enumerate(labels):
        color = theme["series"][index]
        parts.append(
            f'<line x1="{legend_x}" x2="{legend_x + 16}" y1="60" y2="60" stroke="{color}" '
            f'stroke-width="2"/><circle cx="{legend_x + 8}" cy="60" r="4" fill="{color}" '
            f'stroke="{theme["surface"]}" stroke-width="2"/>'
        )
        parts.append(_text(legend_x + 22, 64, f"{label} profile", theme["secondary"], 11))
        legend_x += 130
    for position, (family, title, subtitle) in enumerate(PANELS):
        column, row = position % 2, position // 2
        x = MARGIN + column * (PANEL_W + GAP_X)
        y = HEADER + row * (PANEL_H + GAP_Y)
        parts.append(f'<g transform="translate({x},{y})">')
        parts.append(_panel(reports, family, title, subtitle, theme, labels))
        parts.append("</g>")
    alarms = [report["summary"]["clean_false_alarms"] for report in reports]
    note = "Clean refreshes paged: " + "; ".join(
        f"{label} {item['hits']}/{item['n']}" for label, item in zip(labels, alarms, strict=True)
    )
    parts.append(_text(MARGIN, height - 18, note, theme["secondary"], 11))
    parts.append("</svg>")
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument("--out", required=True, help="output path prefix")
    args = parser.parse_args()
    reports = [json.loads(Path(path).read_text()) for path in (args.first, args.second)]
    prefix = Path(args.out)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for mode in THEMES:
        target = prefix.with_name(f"{prefix.name}-{mode}.svg")
        target.write_text(render(reports, mode))
        print(f"wrote {target}")


if __name__ == "__main__":
    main()
