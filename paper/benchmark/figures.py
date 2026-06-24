"""Render the settled-gate FTC benchmark headline figure from committed evidence.

Reads the same evidence the leaderboard does
(``evidence/program/taxonomy_settled.json`` and ``evidence/program/gain_bias_dob.json``)
and writes a vector PDF, ``paper/benchmark/figs/fig_leaderboard.pdf``, a grouped bar chart
of the settled-science rate on the two controllable continuous fault classes SIGN and GAIN.
Every plotted number traces to a producer's evidence and matches ``benchmark/leaderboard.md``.

Run from repo root:  python paper/benchmark/figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
TAX = json.loads((REPO / "evidence/program/taxonomy_settled.json").read_text())
DOB = json.loads((REPO / "evidence/program/gain_bias_dob.json").read_text())
OUT = REPO / "paper/benchmark/figs/fig_leaderboard.pdf"

# (display label, source, key); source: tax -> taxonomy_settled, dob -> gain_bias_dob.
# Same keys the leaderboard uses, restricted to the requested controllers.
ENTRIES: list[tuple[str, str, str]] = [
    ("PD\n(fault-unaware)", "tax", "pd_fault_unaware"),
    ("Classical\nadaptive", "tax", "classical_adaptive"),
    ("RMA student\n(ours)", "tax", "rma_student_latched"),
    ("RMA + dist.\nobserver (ours)", "dob", "rma_widebias_plus_dob"),
    ("Privileged\noracle", "tax", "teacher_privileged"),
]
CLASSES: list[str] = ["sign", "gain"]
CLASS_LABEL: dict[str, str] = {"sign": "SIGN", "gain": "GAIN"}


def _rate_pct(source: str, key: str, fclass: str) -> float:
    """Return the settled-science rate (%) for one controller on one fault class."""
    if source == "tax":
        rate = TAX["by_class"][fclass][key]["settled_science"]["rate"]
    else:
        rate = DOB["by_class"][fclass][key]["settled_science_rate"]
    return 100.0 * float(rate)


def _collect() -> tuple[list[str], dict[str, list[float]]]:
    """Collect labels and per-class rates, ordered best-to-worst by SIGN+GAIN total."""
    rows = [
        (label, {c: _rate_pct(src, key, c) for c in CLASSES})
        for label, src, key in ENTRIES
    ]
    rows.sort(key=lambda r: r[1]["sign"] + r[1]["gain"], reverse=True)
    labels = [r[0] for r in rows]
    rates = {c: [r[1][c] for r in rows] for c in CLASSES}
    return labels, rates


def _print_table(labels: list[str], rates: dict[str, list[float]]) -> None:
    """Print the exact numbers being plotted so they can be checked against the tables."""
    print("Per-controller settled-science rate (%) to be plotted (best-to-worst):")
    print(f"  {'controller':<22}{'SIGN':>8}{'GAIN':>8}")
    for i, label in enumerate(labels):
        flat = label.replace("\n", " ")
        print(f"  {flat:<22}{rates['sign'][i]:>8.1f}{rates['gain'][i]:>8.1f}")


def make_figure() -> None:
    """Build and save the grouped bar chart as a vector PDF."""
    labels, rates = _collect()
    _print_table(labels, rates)

    plt.rcParams.update({"font.size": 8.5})
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    n = len(labels)
    positions = list(range(n))
    width = 0.38
    colors = {"sign": "#2c6fbb", "gain": "#d1731f"}

    for offset, fclass in ((-width / 2, "sign"), (width / 2, "gain")):
        xs = [p + offset for p in positions]
        bars = ax.bar(
            xs, rates[fclass], width, label=CLASS_LABEL[fclass], color=colors[fclass]
        )
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=7.0)

    ax.set_ylabel("settled-science rate (%)")
    ax.set_ylim(0, 112)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_title("Settled-gate leaderboard (held-out faults)")
    ax.legend(title="fault class", frameon=False, loc="center right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="0.85", linewidth=0.6)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    make_figure()
