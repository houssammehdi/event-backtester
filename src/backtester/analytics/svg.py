"""Small SVG charts for the HTML tear sheet: time-series lines and areas, histograms.

The charts are plain SVG strings styled by CSS custom properties (so the page's light
and dark themes restyle them) and need no plotting library. Every chart is a
``<figure>`` holding the SVG, an HTML legend when there are two or more series and,
for time series, a JSON block that the page's script uses for the crosshair tooltip.
"""

from __future__ import annotations

import html
import itertools
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

FloatArray = npt.NDArray[np.float64]
Formatter = Callable[[float], str]

WIDTH = 960.0
"""Default SVG width in user units (1 unit renders as about 1 px); half-width charts
use about 520."""
LEFT, RIGHT, TOP, BOTTOM = 64.0, 20.0, 14.0, 30.0
END_LABELS = 96.0
"""Extra right margin of time-series charts, for the end-of-line labels."""
MAX_PATH_POINTS = 1600
MAX_TOOLTIP_POINTS = 520


@dataclass(frozen=True, slots=True)
class Series:
    """One line of a time-series chart."""

    name: str
    y: FloatArray
    """Values aligned with the chart's timestamps (NaN leaves a gap)."""
    color: str
    """CSS colour, normally a custom property such as ``var(--series-1)``."""
    area: bool = False
    """Also fill between the line and the chart's baseline with a light wash."""


@dataclass(frozen=True, slots=True)
class Overlay:
    """A density drawn over a histogram, scaled to the histogram's counts."""

    name: str
    pdf: Callable[[FloatArray], FloatArray]
    color: str


def _esc(text: str) -> str:
    return html.escape(text, quote=True)


def _f(value: float) -> str:
    return f"{value:.1f}"


def _hline(cls: str, x1: float, x2: float, y: float) -> str:
    return f'<line class="{cls}" x1="{_f(x1)}" x2="{_f(x2)}" y1="{_f(y)}" y2="{_f(y)}"/>'


def _vline(cls: str, x: float, y1: float, y2: float) -> str:
    return f'<line class="{cls}" x1="{_f(x)}" x2="{_f(x)}" y1="{_f(y1)}" y2="{_f(y2)}"/>'


def _text(cls: str, x: float, y: float, content: str, anchor: str = "start") -> str:
    return (
        f'<text class="{cls}" x="{_f(x)}" y="{_f(y)}" text-anchor="{anchor}">{_esc(content)}</text>'
    )


# ---------------------------------------------------------------------- scales & ticks
def nice_step(span: float, target: int = 5) -> float:
    """Tick step near ``span / target``: 1, 2, 2.5 or 5 times a power of ten."""
    if not math.isfinite(span) or span <= 0:
        return 1.0
    raw = span / max(target, 1)
    magnitude = 10.0 ** math.floor(math.log10(raw))
    return next(m * magnitude for m in (1.0, 2.0, 2.5, 5.0, 10.0) if m * magnitude >= raw * 0.999)


def nice_ticks(lo: float, hi: float, target: int = 5) -> list[float]:
    """Round tick values whose range covers ``[lo, hi]`` (the first and last bracket it)."""
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return [0.0, 1.0]
    if hi <= lo:
        pad = abs(lo) * 0.1 or 1.0
        lo, hi = lo - pad, hi + pad
    step = nice_step(hi - lo, target)
    first, last = math.floor(lo / step + 1e-9), math.ceil(hi / step - 1e-9)
    return [round(k * step, 12) + 0.0 for k in range(first, last + 1)]


def time_ticks(index: pd.DatetimeIndex, max_ticks: int = 11) -> list[tuple[pd.Timestamp, str]]:
    """Calendar ticks: year starts for multi-year spans, month starts otherwise."""
    start, end = index[0], index[-1]
    if (end - start).days >= 2 * 365:
        first, last = start.year + (start.dayofyear > 1), end.year
        step = next(s for s in (1, 2, 5, 10, 20, 50) if (last - first) // s + 1 <= max_ticks)
        return [
            (pd.Timestamp(year=year, month=1, day=1, tz=start.tz), str(year))
            for year in range(first, last + 1, step)
        ]
    months = pd.date_range(start.normalize(), end, freq="MS")
    months = months[months >= start]
    step = next(s for s in (1, 2, 3, 6, 12) if len(months[::s]) <= max_ticks)
    return [
        (ts, ts.strftime("%b %Y") if k == 0 or ts.month == 1 else ts.strftime("%b"))
        for k, ts in enumerate(months[::step])
    ]


class _Linear:
    """Affine map from a data interval onto a pixel interval."""

    def __init__(self, d0: float, d1: float, r0: float, r1: float) -> None:
        self.d0, self.r0 = d0, r0
        self.k = (r1 - r0) / (d1 - d0) if d1 != d0 else 0.0

    def at(self, value: float) -> float:
        return self.r0 + (value - self.d0) * self.k

    def many(self, values: FloatArray) -> FloatArray:
        out: FloatArray = self.r0 + (values - self.d0) * self.k
        return out


def _nanoseconds(index: pd.DatetimeIndex) -> FloatArray:
    """UTC epoch nanoseconds, the unit of ``Timestamp.value``."""
    return index.to_numpy(dtype="datetime64[ns]").astype(np.int64).astype(np.float64)


def _decimate(x: FloatArray, y: FloatArray, buckets: int) -> npt.NDArray[np.intp]:
    """Indices keeping the first, lowest, highest and last point of each x bucket."""
    n = len(x)
    if n <= buckets * 2:
        return np.arange(n)
    edges = np.linspace(x[0], x[-1], buckets + 1)
    which = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, buckets - 1)
    keep = {0, n - 1}
    starts = np.flatnonzero(np.r_[True, which[1:] != which[:-1]])
    stops = np.r_[starts[1:], n]
    for a, b in zip(starts, stops, strict=True):
        keep.update((int(a), int(b - 1)))
        chunk = y[a:b]
        if np.isfinite(chunk).any():
            keep.update((int(a + np.nanargmin(chunk)), int(a + np.nanargmax(chunk))))
    return np.array(sorted(keep), dtype=np.intp)


def _path(xs: FloatArray, ys: FloatArray) -> str:
    """SVG path through the finite points; a NaN ends the current sub-path."""
    parts: list[str] = []
    pen_down = False
    for x, y in zip(xs, ys, strict=True):
        if not math.isfinite(y):
            pen_down = False
            continue
        parts.append(f"{'L' if pen_down else 'M'}{_f(x)},{_f(y)}")
        pen_down = True
    return "".join(parts)


def _legend(entries: Sequence[tuple[str, str, str]]) -> str:
    """HTML legend of ``(name, colour, "line" | "rect")`` - only for two or more."""
    if len(entries) < 2:
        return ""
    items = "".join(
        f'<li><span class="key {shape}" style="--c:{color}"></span>{_esc(name)}</li>'
        for name, color, shape in entries
    )
    return f'<ul class="legend">{items}</ul>'


def _figure(title: str, subtitle: str, body: str, *, legend: str = "", data: str = "") -> str:
    head = f"<figcaption><h3>{_esc(title)}</h3>"
    if subtitle:
        head += f'<p class="sub">{_esc(subtitle)}</p>'
    head += "</figcaption>"
    script = f'<script type="application/json" class="chart-data">{data}</script>' if data else ""
    return f'<figure class="chart">{head}{legend}<div class="plot">{body}</div>{script}</figure>'


def _json(payload: object) -> str:
    return json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")


def _svg_open(title: str, width: float, height: float, *, focusable: bool) -> str:
    focus = ' tabindex="0"' if focusable else ""
    box = f"0 0 {width:.0f} {height:.0f}"
    size = f"min-width:{0.75 * width:.0f}px"  # below that, the card scrolls sideways
    return f'<svg viewBox="{box}" style="{size}" role="img"{focus} aria-label="{_esc(title)}">'


def _x_label(x: float, y: float, content: str, x_min: float, x_max: float) -> str:
    """An x-axis tick label, anchored inwards near the ends so it is never clipped."""
    anchor = "end" if x > x_max - 24 else "start" if x < x_min + 24 else "middle"
    return _text("tick", x, y, content, anchor)


# ---------------------------------------------------------------------- time series
def time_series_chart(
    title: str,
    index: pd.DatetimeIndex,
    series: Sequence[Series],
    *,
    y_format: Formatter,
    height: float = 260.0,
    subtitle: str = "",
    baseline: float | None = None,
    include_zero: bool = False,
    width: float = WIDTH,
    tick_format: Formatter | None = None,
) -> str:
    """A line chart over time on one y-axis, with a crosshair tooltip and end labels.

    Args:
        title: Chart title.
        index: Timestamps shared by every series.
        series: Lines to draw, in legend order.
        y_format: Formats tick labels, end labels and tooltip values.
        height: SVG height in user units.
        subtitle: Line under the title.
        baseline: Value drawn as the axis line (and the floor of area fills);
            defaults to the lowest tick.
        include_zero: Extend the y-range to include zero.
        width: SVG width in user units.
        tick_format: Formats the y-axis ticks (default: ``y_format``).
    """
    x_num = _nanoseconds(index)
    finite = [s.y[np.isfinite(s.y)] for s in series]
    values = np.concatenate([f for f in finite if len(f)] or [np.zeros(1)])
    lo, hi = float(values.min()), float(values.max())
    if include_zero or baseline is not None:
        ref = 0.0 if baseline is None else baseline
        lo, hi = min(lo, ref), max(hi, ref)
    ticks = nice_ticks(lo, hi)
    x_end = width - RIGHT - END_LABELS
    sx = _Linear(x_num[0], x_num[-1], LEFT, x_end)
    sy = _Linear(ticks[0], ticks[-1], height - BOTTOM, TOP)
    parts = [_svg_open(title, width, height, focusable=True)]
    for t in ticks:
        y = sy.at(t)
        parts += [
            _hline("grid", LEFT, x_end, y),
            _text("tick", LEFT - 8, y + 4, (tick_format or y_format)(t), "end"),
        ]
    for ts, label in time_ticks(index, max_ticks=11 if width >= 800 else 6):
        x = sx.at(float(ts.value))
        if LEFT <= x <= x_end:
            parts.append(_x_label(x, height - 8, label, LEFT, x_end))
    y_base = sy.at(ticks[0] if baseline is None else baseline)
    parts.append(_hline("axis", LEFT, x_end, y_base))
    buckets = int(min(MAX_PATH_POINTS // 2, x_end - LEFT))
    labels: list[tuple[float, str, str]] = []
    for s in series:
        keep = _decimate(x_num, s.y, buckets)
        xs, ys = sx.many(x_num[keep]), sy.many(s.y[keep])
        line = _path(xs, ys)
        if not line:
            continue
        if s.area:
            ok = np.flatnonzero(np.isfinite(ys))
            x0, x1 = _f(xs[ok[0]]), _f(xs[ok[-1]])
            area = f"M{x0},{_f(y_base)}{line.replace('M', 'L')}L{x1},{_f(y_base)}Z"
            parts.append(f'<path class="area" d="{area}" style="fill:{s.color}"/>')
        parts.append(f'<path class="line" d="{line}" style="stroke:{s.color}"/>')
        last = float(s.y[np.flatnonzero(np.isfinite(s.y))[-1]])
        labels.append((sy.at(last), s.name, y_format(last)))
    labels.sort()
    if all(b[0] - a[0] >= 26 for a, b in itertools.pairwise(labels)):  # else: legend only
        for y, name, text in labels:
            parts += [
                _text("end", x_end + 8, y - 3, text),
                _text("end-name", x_end + 8, y + 11, name),
            ]
    parts.append('<g class="hover" visibility="hidden"><line class="cross"/></g></svg>')
    step = max(1, math.ceil(len(index) / MAX_TOOLTIP_POINTS))
    sample = np.unique(np.r_[np.arange(0, len(index), step), len(index) - 1])
    payload = {
        "x": [round(v, 1) for v in sx.many(x_num[sample]).tolist()],
        "top": TOP,
        "bottom": height - BOTTOM,
        "labels": [ts.strftime("%Y-%m-%d") for ts in index[sample]],
        "series": [
            {
                "name": s.name,
                "color": s.color,
                "y": [round(v, 1) if math.isfinite(v) else None for v in sy.many(s.y[sample])],
                "text": [y_format(v) if math.isfinite(v) else "n/a" for v in s.y[sample].tolist()],
            }
            for s in series
        ],
    }
    legend = _legend([(s.name, s.color, "line") for s in series])
    return _figure(title, subtitle, "".join(parts), legend=legend, data=_json(payload))


# ---------------------------------------------------------------------- histograms
def _bar(x: float, width: float, y_base: float, y_top: float, radius: float = 4.0) -> str:
    """Path of a bar with rounded corners at its data end, square at the baseline."""
    height = abs(y_base - y_top)
    r = min(radius, width / 2, height)
    d = 1.0 if y_top < y_base else -1.0  # from the data end towards the baseline
    x1 = x + width
    return (
        f"M{_f(x)},{_f(y_base)}L{_f(x)},{_f(y_top + d * r)}"
        f"Q{_f(x)},{_f(y_top)} {_f(x + r)},{_f(y_top)}L{_f(x1 - r)},{_f(y_top)}"
        f"Q{_f(x1)},{_f(y_top)} {_f(x1)},{_f(y_top + d * r)}L{_f(x1)},{_f(y_base)}Z"
    )


def _bin_edges(lo: float, hi: float, bins: int, *, at_zero: bool) -> FloatArray:
    if at_zero and lo < 0 < hi:  # one edge exactly at zero, so bins have one sign
        width = max(-lo, hi) / max(1, bins // 2)
        k = np.arange(math.floor(lo / width), math.ceil(hi / width) + 1, dtype=np.float64)
        edges: FloatArray = k * width
        return edges
    return np.linspace(lo, hi, bins + 1)


def histogram_chart(
    title: str,
    values: FloatArray,
    *,
    x_format: Formatter,
    bins: int = 40,
    subtitle: str = "",
    color: str = "var(--series-1)",
    name: str = "Frequency",
    diverging: bool = False,
    overlay: Overlay | None = None,
    marks: Sequence[tuple[float, str]] = (),
    height: float = 240.0,
    unit: str = "observations",
    width: float = WIDTH,
) -> str:
    """Histogram with a 2px surface gap between bins and a tooltip on every bar.

    With ``diverging`` the bins below zero take the negative colour and those above
    zero the positive one (a bin edge sits at zero). ``marks`` are labelled vertical
    reference lines, such as the bounds of an interval.
    """
    v = values[np.isfinite(values)]
    if len(v) == 0:
        return _figure(title, subtitle, '<p class="empty">Nothing to show.</p>')
    lo, hi = float(v.min()), float(v.max())
    if hi == lo:
        lo, hi = lo - 0.5 * (abs(lo) or 1.0), hi + 0.5 * (abs(hi) or 1.0)
    counts, edges = np.histogram(v, bins=_bin_edges(lo, hi, bins, at_zero=diverging))
    curve_x = np.linspace(float(edges[0]), float(edges[-1]), 240)
    curve_y = np.zeros_like(curve_x)
    top = float(counts.max())
    if overlay is not None:
        curve_y = len(v) * float(edges[1] - edges[0]) * overlay.pdf(curve_x)
        top = max(top, float(np.nanmax(curve_y)))
    y_ticks = nice_ticks(0.0, top, 4)
    x_end = width - RIGHT
    sx = _Linear(float(edges[0]), float(edges[-1]), LEFT, x_end)
    sy = _Linear(0.0, y_ticks[-1], height - BOTTOM, TOP)
    y_base = sy.at(0.0)
    parts = [_svg_open(title, width, height, focusable=False)]
    for t in y_ticks:
        y = sy.at(t)
        parts += [
            _hline("grid", LEFT, x_end, y),
            _text("tick", LEFT - 8, y + 4, f"{t:,.0f}", "end"),
        ]
    for t in nice_ticks(float(edges[0]), float(edges[-1]), 8 if width >= 800 else 5):
        x = sx.at(t)
        if LEFT - 0.5 <= x <= x_end + 0.5:
            parts.append(_x_label(x, height - 8, x_format(t), LEFT, x_end))
    for count, a, b in zip(counts.tolist(), edges[:-1].tolist(), edges[1:].tolist(), strict=True):
        if count == 0:
            continue
        x0, x1 = sx.at(a) + 1.0, sx.at(b) - 1.0  # the 2px surface gap between bins
        fill = ("var(--negative)" if b <= 0 else "var(--positive)") if diverging else color
        noun = unit if count != 1 else unit.removesuffix("s")
        tip = f"{x_format(a)} to {x_format(b)}: {count:,} {noun}"
        path = _bar(x0, max(x1 - x0, 0.5), y_base, sy.at(count))
        parts.append(
            f'<path class="bar" tabindex="0" d="{path}" style="fill:{fill}" '
            f'data-tip="{_esc(tip)}"/>'
        )
    if overlay is not None:
        curve = _path(sx.many(curve_x), sy.many(curve_y))
        parts.append(f'<path class="line" d="{curve}" style="stroke:{overlay.color}"/>')
    for value, label in marks:
        x = sx.at(value)
        if LEFT <= x <= x_end:
            parts += [_vline("mark", x, TOP, y_base), _text("mark-label", x + 4, TOP + 10, label)]
    parts.append(_hline("axis", LEFT, x_end, y_base))
    parts.append("</svg>")
    entries = [(name, color, "rect")]
    if diverging:
        entries = [("Losses", "var(--negative)", "rect"), ("Gains", "var(--positive)", "rect")]
    if overlay is not None:
        entries.append((overlay.name, overlay.color, "line"))
    return _figure(title, subtitle, "".join(parts), legend=_legend(entries))
