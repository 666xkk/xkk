#!/usr/bin/env python3
"""Generate the IEEE TDSC-style overview of the Orthrus incident pipeline."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


DEFAULT_OUT = Path("paper1/figures/final_4dataset_TDSC")

BLUE = "#2878B5"
LIGHT_BLUE = "#DCECF7"
ORANGE = "#F28E2B"
LIGHT_ORANGE = "#FCE6CF"
GREEN = "#59A14F"
LIGHT_GREEN = "#DDEED9"
PURPLE = "#7B61A8"
LIGHT_PURPLE = "#E9E0F3"
GRAY = "#5F6670"
LIGHT_GRAY = "#F2F4F6"
DARK = "#20242A"
RED = "#D9534F"


def rounded(ax, xy, width, height, fc, ec=GRAY, lw=1.3, radius=0.018, z=1):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        transform=ax.transAxes,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, color=GRAY, lw=1.7, style="-|>", z=5, connection=None):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=13,
        linewidth=lw,
        color=color,
        transform=ax.transAxes,
        connectionstyle=connection,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def label(ax, x, y, text, size=10, weight="normal", color=DARK, ha="center", va="center", **kw):
    return ax.text(
        # The canvas is 16 in wide but is inserted at IEEE's 7.16 in text width.
        # Scaling here keeps the final printed text approximately 7--12 pt.
        x, y, text, transform=ax.transAxes, fontsize=size * 1.30, fontweight=weight,
        color=color, ha=ha, va=va, zorder=10, **kw
    )


def draw_provenance(ax, x, y, w, h):
    nodes = {
        "P1": (x + 0.20*w, y + 0.66*h, BLUE),
        "P2": (x + 0.22*w, y + 0.25*h, BLUE),
        "F1": (x + 0.70*w, y + 0.75*h, ORANGE),
        "F2": (x + 0.77*w, y + 0.30*h, ORANGE),
        "N": (x + 0.50*w, y + 0.50*h, GREEN),
    }
    edges = [("P1", "F1"), ("P1", "N"), ("N", "F2"), ("P2", "F2"), ("P2", "N")]
    for left, right in edges:
        p1, p2 = nodes[left], nodes[right]
        arrow(ax, p1[:2], p2[:2], color="#8A929B", lw=1.1, z=3)
    for name, (nx, ny, color) in nodes.items():
        ax.add_patch(Circle((nx, ny), 0.017, transform=ax.transAxes,
                            facecolor=color, edgecolor="white", linewidth=1.2, zorder=6))
        label(ax, nx, ny - 0.035, name, size=7.3, color=GRAY)


def draw_mini_series(ax, x, y, w, h, values, color, selected):
    values = np.asarray(values, dtype=float)
    values = (values - values.min()) / max(values.max() - values.min(), 1e-9)
    xs = np.linspace(x, x + w, len(values))
    ys = y + 0.10*h + values * 0.75*h
    ax.plot(xs, ys, color=color, linewidth=1.7, transform=ax.transAxes, zorder=6)
    ax.fill_between(xs, y + 0.10*h, ys, color=color, alpha=0.10,
                    transform=ax.transAxes, zorder=4)
    for index in selected:
        ax.scatter([xs[index]], [ys[index]], s=28, color=RED, edgecolor="white",
                   linewidth=0.7, transform=ax.transAxes, zorder=8)


def configure_tdsc_style():
    """Use embedded, IEEE-friendly serif fonts and restrained line weights."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def main(output_dir=DEFAULT_OUT):
    configure_tdsc_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Title and one-line claim.
    label(ax, 0.035, 0.965,
          "Causal Multiscale Incident Reconstruction on a Frozen Orthrus Detector",
          size=19, weight="bold", ha="left")
    label(ax, 0.035, 0.925,
          "Normal-only learning  •  validation-selected checkpoint  •  strictly causal test-time aggregation  •  labels used only after ranking",
          size=10.5, color=GRAY, ha="left")

    # Protocol strip.
    rounded(ax, (0.035, 0.825), 0.93, 0.072, LIGHT_GRAY, ec="#C5CBD2", lw=1.0)
    stages = [
        (0.06, 0.18, LIGHT_BLUE, BLUE, "1  Train", "normal train windows"),
        (0.275, 0.18, LIGHT_GREEN, GREEN, "2  Select", "minimum validation loss"),
        (0.49, 0.20, LIGHT_PURPLE, PURPLE, "3  Freeze", "checkpoint + parameters"),
        (0.725, 0.205, LIGHT_ORANGE, ORANGE, "4  Infer", "chronological test windows"),
    ]
    for sx, sw, fc, ec, title, subtitle in stages:
        rounded(ax, (sx, 0.839), sw, 0.044, fc, ec=ec, lw=1.0, radius=0.01)
        label(ax, sx + 0.012, 0.867, title, size=9.4, weight="bold", color=ec, ha="left")
        label(ax, sx + 0.012, 0.850, subtitle, size=7.5, color=GRAY, ha="left")
    for sx in (0.245, 0.46, 0.705):
        arrow(ax, (sx, 0.861), (sx + 0.022, 0.861), lw=1.2)

    # Three large method regions.
    rounded(ax, (0.035, 0.185), 0.275, 0.595, "#FBFCFD", ec=BLUE, lw=1.7)
    rounded(ax, (0.33, 0.185), 0.39, 0.595, "#FFFDFC", ec=ORANGE, lw=1.7)
    rounded(ax, (0.74, 0.185), 0.225, 0.595, "#FCFBFE", ec=PURPLE, lw=1.7)

    label(ax, 0.052, 0.754, "I. Frozen edge-anomaly backbone", size=12.5,
          weight="bold", color=BLUE, ha="left")
    label(ax, 0.347, 0.754, "II. Strictly causal multiscale evidence", size=12.5,
          weight="bold", color=ORANGE, ha="left")
    label(ax, 0.757, 0.754, "III. Incident reconstruction", size=12.5,
          weight="bold", color=PURPLE, ha="left")

    # Region I.
    rounded(ax, (0.055, 0.57), 0.105, 0.13, LIGHT_GRAY, ec="#A7AFB8", radius=0.012)
    draw_provenance(ax, 0.062, 0.58, 0.09, 0.10)
    label(ax, 0.107, 0.548, "Provenance window $G_t$", size=8.8, weight="bold")

    arrow(ax, (0.166, 0.635), (0.188, 0.635), color=BLUE)
    rounded(ax, (0.19, 0.565), 0.098, 0.14, LIGHT_BLUE, ec=BLUE, radius=0.012)
    label(ax, 0.239, 0.678, "Orthrus", size=11, weight="bold", color=BLUE)
    label(ax, 0.239, 0.647, "node/text embeddings", size=7.8)
    label(ax, 0.239, 0.620, "temporal GNN memory", size=7.8)
    label(ax, 0.239, 0.593, "edge reconstruction", size=7.8)
    label(ax, 0.278, 0.691, "FROZEN", size=5.8, weight="bold", color=BLUE)

    rounded(ax, (0.055, 0.39), 0.233, 0.105, "white", ec="#A7AFB8", radius=0.012)
    label(ax, 0.171, 0.467, "Frozen edge anomaly losses", size=9.5, weight="bold")
    label(ax, 0.171, 0.432, r"$\ell_{e,t}=\mathcal{L}(\hat{x}_{e,t},x_{e,t})$", size=12)
    label(ax, 0.171, 0.405, "one loss per test edge; no parameter update", size=7.8, color=GRAY)
    arrow(ax, (0.239, 0.562), (0.239, 0.502), color=BLUE)

    rounded(ax, (0.055, 0.245), 0.233, 0.095, LIGHT_BLUE, ec=BLUE, radius=0.012)
    label(ax, 0.171, 0.316, "Role-aware node evidence (file role)", size=9.2, weight="bold")
    label(ax, 0.171, 0.282, r"$a_{v,t}=\max_{e\ni v}\ell_{e,t}$", size=11.5)
    label(ax, 0.171, 0.257, "source + destination incidence", size=7.8, color=GRAY)
    arrow(ax, (0.171, 0.388), (0.171, 0.345), color=BLUE)

    # Arrow between regions.
    arrow(ax, (0.306, 0.46), (0.338, 0.46), color=ORANGE, lw=2.2)

    # Region II: collective evidence and two scales.
    rounded(ax, (0.352, 0.61), 0.345, 0.085, LIGHT_ORANGE, ec=ORANGE, radius=0.012)
    label(ax, 0.524, 0.674, "Collective weak-evidence score", size=10, weight="bold", color="#B55B00")
    label(ax, 0.524, 0.642, r"$r_t=\mathrm{mean}\left(\mathrm{TopK}_{v}\;a_{v,t}\right),\quad K=10$", size=12)
    label(ax, 0.524, 0.619, "suppresses isolated edges; retains coordinated weak anomalies", size=7.7, color=GRAY)
    arrow(ax, (0.524, 0.608), (0.524, 0.573), color=ORANGE)

    # Scale branches.
    rounded(ax, (0.352, 0.39), 0.158, 0.175, "white", ec=GREEN, radius=0.012)
    label(ax, 0.431, 0.542, "Scale 1 · immediate", size=9.5, weight="bold", color=GREEN)
    draw_mini_series(ax, 0.37, 0.455, 0.123, 0.065,
                     [1, 2, 1.5, 4.8, 1.2, 2.4, 5.5, 1.3], GREEN, [3, 6])
    label(ax, 0.431, 0.420, r"$c_t^{(1)}=r_t$", size=10.5)
    label(ax, 0.431, 0.399, "Top-20 windows", size=7.8, color=GRAY)

    rounded(ax, (0.539, 0.39), 0.158, 0.175, "white", ec=ORANGE, radius=0.012)
    label(ax, 0.618, 0.542, "Scale 4 · persistent", size=9.5, weight="bold", color=ORANGE)
    draw_mini_series(ax, 0.557, 0.455, 0.123, 0.065,
                     [1, 1.4, 1.8, 3.0, 2.6, 3.2, 3.9, 2.8], ORANGE, [3, 5, 6])
    label(ax, 0.618, 0.420, r"$c_t^{(4)}=\frac{1}{|H_t|}\sum_{j=0}^{3}r_{t-j}$", size=10.2)
    label(ax, 0.618, 0.399, "Top-20 · past only", size=7.8, color=GRAY)

    arrow(ax, (0.431, 0.387), (0.492, 0.338), color=GREEN,
          connection="arc3,rad=-0.15")
    arrow(ax, (0.618, 0.387), (0.555, 0.338), color=ORANGE,
          connection="arc3,rad=0.15")

    rounded(ax, (0.385, 0.235), 0.28, 0.105, LIGHT_PURPLE, ec=PURPLE, radius=0.012)
    label(ax, 0.525, 0.316, "Frozen multiscale union", size=9.8, weight="bold", color=PURPLE)
    label(ax, 0.525, 0.285, r"$\mathcal{W}=\mathrm{Top20}(c^{(1)})\cup\mathrm{Top20}(c^{(4)})$", size=10.6)
    label(ax, 0.525, 0.258, r"$m_t=\max_s\,p_t^{(s)}$  ·  rank-percentile fusion", size=8.7, color=GRAY)

    # Arrow to region III.
    arrow(ax, (0.716, 0.46), (0.748, 0.46), color=PURPLE, lw=2.2)

    # Region III: clustering and analyst output.
    rounded(ax, (0.765, 0.565), 0.175, 0.135, LIGHT_PURPLE, ec=PURPLE, radius=0.012)
    label(ax, 0.852, 0.675, "Causal window clustering", size=10, weight="bold", color=PURPLE)
    # Small window ribbon.
    for i in range(9):
        selected = i in {1, 2, 4, 7, 8}
        ax.add_patch(Rectangle(
            (0.785 + i*0.0155, 0.617), 0.0115, 0.026,
            transform=ax.transAxes,
            facecolor=PURPLE if selected else "white",
            edgecolor="#A9A0B8", linewidth=0.7, zorder=6,
        ))
    label(ax, 0.852, 0.592, r"same incident if $\Delta t\leq2$ windows", size=8.4)
    label(ax, 0.852, 0.573, "bridges one missing window", size=7.6, color=GRAY)

    arrow(ax, (0.852, 0.56), (0.852, 0.517), color=PURPLE)
    rounded(ax, (0.765, 0.405), 0.175, 0.105, "white", ec=PURPLE, radius=0.012)
    label(ax, 0.852, 0.482, "Incident scoring", size=9.8, weight="bold", color=PURPLE)
    label(ax, 0.852, 0.450, r"$S(I)=\mathrm{mean}\left(\mathrm{Top3}_{t\in I}\;m_t\right)$", size=10.4)
    label(ax, 0.852, 0.422, "rank by score, support, then time", size=7.7, color=GRAY)

    arrow(ax, (0.852, 0.40), (0.852, 0.36), color=PURPLE)
    rounded(ax, (0.765, 0.235), 0.175, 0.118, LIGHT_GREEN, ec=GREEN, radius=0.012)
    label(ax, 0.852, 0.329, "Ranked analyst queue", size=10, weight="bold", color=GREEN)
    for rank, (score, color) in enumerate([(0.96, RED), (0.82, ORANGE), (0.69, BLUE)], 1):
        yy = 0.304 - (rank-1)*0.026
        label(ax, 0.785, yy, f"{rank}", size=8.2, weight="bold", color=color)
        ax.add_patch(Rectangle((0.802, yy-0.007), 0.105*score, 0.014,
                               transform=ax.transAxes, facecolor=color,
                               edgecolor="none", alpha=0.80, zorder=6))
    label(ax, 0.852, 0.247, "30–34 windows → 10–17 incidents", size=7.7, color=GRAY)

    # Bottom guarantees / evaluation boundary.
    rounded(ax, (0.035, 0.055), 0.93, 0.085, "#FFF9EE", ec="#D8B56D", lw=1.1)
    label(ax, 0.055, 0.116, "Operational guarantees", size=9.5, weight="bold", color="#8A5B00", ha="left")
    guarantees = [
        "No future windows",
        "No attack labels in scoring",
        "Fixed K, scales, budgets and gap",
        "Deterministic tie-breaking",
        "Labels enter only post-hoc evaluation",
    ]
    xs = [0.07, 0.245, 0.43, 0.66, 0.835]
    for xx, text in zip(xs, guarantees):
        ax.add_patch(Circle((xx-0.012, 0.084), 0.005, transform=ax.transAxes,
                            facecolor=GREEN, edgecolor="none", zorder=8))
        label(ax, xx, 0.084, text, size=7.8, color=GRAY, ha="left")

    # Evaluation-only label arrow outside computational flow.
    label(ax, 0.962, 0.163, "attack intervals", size=7.4, color=RED, ha="right")
    label(ax, 0.962, 0.146, "evaluation only", size=7.4, color=RED, ha="right")
    arrow(ax, (0.93, 0.157), (0.885, 0.185), color=RED, lw=1.1,
          style="-|>", connection="arc3,rad=-0.2")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    for ext in ("pdf", "svg"):
        fig.savefig(output_dir / f"fig0_method_overview.{ext}", bbox_inches="tight")
    fig.savefig(output_dir / "fig0_method_overview.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    print("generated:")
    for ext in ("pdf", "svg", "png"):
        print(" ", output_dir / f"fig0_method_overview.{ext}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    main(parser.parse_args().output_dir)
