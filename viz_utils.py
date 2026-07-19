"""LinkedIn-ready chart helpers shared by the evaluation scripts.

Every eval script calls one of these to save a clean PNG next to its markdown
report, so each run produces a visual you can post and read at a glance. All
charts share one palette and style, so they look like a coherent set.

Dependency: matplotlib only (no numpy), and every call is meant to be wrapped in
try/except by the caller so a plotting hiccup never breaks an evaluation run.
"""

import matplotlib
matplotlib.use("Agg")  # write straight to file; no display needed
import matplotlib.pyplot as plt

# Shared palette
COLOR_BASE = "#94a3b8"    # slate  = base model ("before")
COLOR_FT = "#2563eb"      # blue   = fine-tuned model ("after")
COLOR_GOOD = "#16a34a"    # green  = improvement / grounded
COLOR_WARN = "#b91c1c"    # red    = fabrication / worse


def _apply_style():
    plt.rcParams.update({
        "font.size": 12,
        "axes.edgecolor": "#cccccc",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def _label_bars(ax, bars, fmt="{:.2f}"):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")


def _finish(fig, ax, subtitle, footer):
    if subtitle:
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha="center",
                fontsize=11, color="#555555")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if footer:
        fig.text(0.5, 0.01, footer, ha="center", fontsize=10, color="#777777")
    fig.tight_layout(rect=[0, 0.03, 1, 1])


def grouped_bar(out_path, title, group_labels, series, *, ylabel="",
                subtitle="", footer="", value_fmt="{:.2f}", colors=None, ymax=None):
    """Grouped/side-by-side bar chart.

    series: dict {series_name: [one value per group]}. One cluster per group label,
    one bar per series. Value labels are drawn on every bar.
    """
    _apply_style()
    n_groups = len(group_labels)
    n_series = len(series)
    x = list(range(n_groups))
    width = 0.8 / max(n_series, 1)
    fig, ax = plt.subplots(figsize=(max(7, 1.9 * n_groups + 2), 6))
    palette = colors or [COLOR_BASE, COLOR_FT, COLOR_GOOD, COLOR_WARN]
    for i, (name, vals) in enumerate(series.items()):
        offset = (i - (n_series - 1) / 2) * width
        positions = [xi + offset for xi in x]
        bars = ax.bar(positions, vals, width, label=name, color=palette[i % len(palette)])
        _label_bars(ax, bars, value_fmt)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.set_ylabel(ylabel)
    if ymax:
        ax.set_ylim(0, ymax)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=28 if subtitle else 14)
    if n_series > 1:
        ax.legend(loc="best", frameon=True)
    _finish(fig, ax, subtitle, footer)
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    return out_path


def dual_panel(out_path, title, panels, *, subtitle="", footer="", colors=None):
    """Two side-by-side bar panels for metrics on different scales (e.g. latency).

    panels: list of exactly 2 dicts, each:
        {"label": str, "value_fmt": str, "series": {name: value}}
    """
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    palette = colors or [COLOR_BASE, COLOR_FT]
    for ax, panel in zip(axes, panels):
        names = list(panel["series"].keys())
        vals = list(panel["series"].values())
        bars = ax.bar(names, vals, color=[palette[i % len(palette)] for i in range(len(names))],
                      width=0.55)
        _label_bars(ax, bars, panel.get("value_fmt", "{:.2f}"))
        ax.set_title(panel["label"], fontsize=12.5, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    if subtitle:
        fig.text(0.5, 0.92, subtitle, ha="center", fontsize=11, color="#555555")
    if footer:
        fig.text(0.5, 0.01, footer, ha="center", fontsize=10, color="#777777")
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    return out_path


def chart_path_for(report_path: str) -> str:
    """Given 'foo/metrics_report.md', return 'foo/metrics_report.png'."""
    import os
    base, _ = os.path.splitext(report_path)
    return base + ".png"
