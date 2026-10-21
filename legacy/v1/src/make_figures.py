#!/usr/bin/env python3
"""Reproduce four teaching figures for the simple CGP model.

Run from any directory with:
    python3 /path/to/CGP_Model_Figures/src/make_figures.py

Outputs are placed beside this script's parent folder. All values are
illustrative, not fitted to observations. The CSV files contain the actual
values plotted. The PDF guide pairs each vector figure with its argument,
mathematical interpretation and scope. See README.txt for the model.
"""

from pathlib import Path
from io import BytesIO
import csv
import json
import math
import textwrap
import zipfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
from scipy.optimize import brentq
from pypdf import PdfReader, PdfWriter, Transformation
from reportlab.pdfgen import canvas
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
DATA = ROOT / "data"
for folder in (FIGURES, DATA):
    folder.mkdir(parents=True, exist_ok=True)

# The two action colours stay the same in Figures 2 and 3. Other figures
# identify their own trajectories explicitly with both colour and line style.
BLUE = "#246879"
ORANGE = "#BE652C"
PURPLE = "#76558B"
INK = "#24333B"
GREY = "#8A9499"
PALE_GREY = "#DEE3E5"
GRID = "#E9EDEF"
HALF_COLOURS = ["#76A9B3", "#367F91", "#174F62"]
HALF_STYLES = ["--", "-", "-."]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "semibold",
    "axes.edgecolor": "#879298",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "lines.linewidth": 2.6,
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42,
    # Editable text is retained in SVG rather than converted to outlines.
    "svg.fonttype": "none",
    "savefig.facecolor": "white",
})


def remaining(initial, age, half_life):
    """Functional availability, not necessarily the fraction alive in storage."""
    return np.asarray(initial) * np.exp2(-np.asarray(age) / half_life)


def fertility(amount):
    """Assumed saturating fertilisation response: amount=1 gives probability 1/2."""
    return np.asarray(amount) / (1.0 + np.asarray(amount))


def payoffs(x, y, w, cost):
    """Net payoff per potential egg over the next short reproductive bout.

    Old conspecific sperm remain available after the new ejaculate arrives.
    Sperm use is proportional to functional availability. Hybrid fitness w
    weights the outcome after fertilisation, not the probability of paternity.
    """
    x = np.asarray(x)
    return fertility(x), (x + w * y) / (1.0 + x + y) - cost


def acceptance_boundary(x, y, cost):
    """The hybrid fitness w at which accepting and rejecting have equal payoff.

    Rearranging Delta W = y/(1+x+y) * [w-x/(1+x)] - cost gives
    w_crit = x/(1+x) + cost*(1+x+y)/y. Values above one mean that
    no w in the chosen range [0,1] makes acceptance beneficial.
    """
    return fertility(x) + cost * (1.0 + np.asarray(x) + y) / y


def cost_free_switch(x0, half_life, w):
    """Earliest nonnegative switch time for 0<w<1, y>0 and zero mating cost."""
    return np.maximum(0.0, half_life * np.log2(x0 * (1.0 - w) / w))


def csv_output(name, fields, rows):
    with (DATA / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)


def new_figure(title, subtitle, equation, footer):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5))
    fig.subplots_adjust(left=0.085, right=0.97, bottom=0.20, top=0.73, wspace=0.34)
    fig.text(0.055, 0.965, title, fontsize=19, fontweight="semibold", va="top")
    fig.text(0.055, 0.906, subtitle, fontsize=10.8, va="top", color="#536168")
    fig.text(0.5, 0.839, equation, fontsize=16, ha="center", va="center")
    fig.text(0.055, 0.038, footer, fontsize=9.5, va="bottom", color="#536168")
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=GRID, linewidth=0.7)
        ax.tick_params(length=3, pad=5)
    return fig, axes


def panel(ax, title):
    ax.set_title(title, loc="left", pad=17, fontsize=12)


def save_figure(fig, stem):
    # Matching vector and raster exports allow paper, presentation and web use.
    for extension in ("pdf", "svg", "png"):
        fig.savefig(FIGURES / f"{stem}.{extension}", dpi=300)
    plt.close(fig)


CAPTIONS = []


def figure_1():
    t = np.linspace(0, 8, 401)
    fig, (a, b) = new_figure(
        "1  Functional lifespan and initial amount are different traits",
        "A larger starting store can preserve availability without changing the rate of functional loss.",
        r"$S(t)=2^{-t/h}$     ;     $x(t)=x_0 S(t)$",
        "Illustrative model. A: equal initial cohorts. B: a common functional half-life of 2 days.",
    )
    panel(a, "A  Longer half-life slows functional loss")
    rows = []
    for h, col, style in zip([1, 2, 4], HALF_COLOURS, HALF_STYLES):
        s = remaining(1, t, h)
        a.plot(t, s, color=col, ls=style, label=f"Half-life: {h} day" + ("s" if h > 1 else ""))
        a.vlines(h, 0, 0.5, color=col, linewidth=1, linestyles=":")
        a.plot(h, 0.5, "o", color=col, markerfacecolor="white", markersize=6, markeredgewidth=1.7)
        rows.extend(zip(t, np.full(t.size, h), s))
    a.axhline(0.5, color="#A8B1B5", ls=":", lw=1, zorder=0)
    a.text(7.8, 0.52, "Half remains", ha="right", fontsize=9, color="#69767C")
    a.set(xlim=(0, 8), ylim=(0, 1.04), xlabel="Age of the functional cohort (days)",
          ylabel="Fraction of the initial functional store")
    a.set_xticks([0, 2, 4, 6, 8])
    a.set_yticks([0, 0.25, 0.5, 0.75, 1])
    a.legend(frameon=False, fontsize=9.5, loc="upper right")
    csv_output("figure_1A_persistence.csv", ["age_days", "half_life_days", "fraction_initial_functional_store"], rows)

    panel(b, "B  A larger initial store raises availability")
    rows = []
    for x0, col, style in zip([2, 4, 8], HALF_COLOURS, HALF_STYLES):
        x = remaining(x0, t, 2)
        b.plot(t, x, color=col, ls=style, label=rf"Initial store $x_0={x0}$")
        rows.extend(zip(t, np.full(t.size, x0), x))
    b.set(xlim=(0, 8), ylim=(0, 8.4), xlabel="Time since first mating (days)",
          ylabel="Functional sperm availability (scaled units)")
    b.set_xticks([0, 2, 4, 6, 8])
    b.set_yticks([0, 2, 4, 6, 8])
    b.legend(frameon=False, fontsize=9.5, loc="upper right")
    csv_output("figure_1B_initial_store.csv", ["age_days", "initial_store_scaled", "functional_availability_scaled"], rows)
    stem = "Figure_1_Persistence_and_Initial_Amount"
    save_figure(fig, stem)
    CAPTIONS.append((stem,
        "Argument: measure lifespan and initial amount separately.",
        "Panel A holds initial availability constant and varies the functional half-life. The open circles mark the age at which half of each initial cohort remains functional and available. Panel B holds the half-life at two days and varies the initial store. Increasing the starting amount raises the entire availability curve without changing proportional decay. In this model, producing more functional sperm and preserving existing sperm are distinct ways to maintain availability. A larger store is a possible consequence of greater production, not a direct measurement of production.",
        "Measurement: S(t) is the surviving functional fraction of the original store, not the fraction alive among sperm retained at a later sample. Initial transfer, entry into storage, death, loss of function, use and ejection require separate measurements. These trajectories do not identify the biological cause of loss."))


def figure_2():
    x0, h, y, w, cost = 4.0, 2.0, 4.0, 0.5, 0.05
    t = np.linspace(0, 10, 501)
    x = remaining(x0, t, h)
    reject, accept = payoffs(x, y, w, cost)
    switch = brentq(lambda age: float(payoffs(remaining(x0, age, h), y, w, cost)[1]
                                         - payoffs(remaining(x0, age, h), y, w, cost)[0]), 0, 10)
    fig, (a, b) = new_figure(
        "2  Remating can help after the conspecific store declines",
        "New sperm can reduce fertilisation failure while also replacing conspecific fertilisations with hybrid fertilisations.",
        r"$W_{\rm reject}=\frac{x}{1+x}$     ;     $W_{\rm accept}=\frac{x+wy}{1+x+y}-c$",
        r"Illustrative values: $x_0=4$, $h_C=2$ days, $y=4$, $w=0.5$, $c=0.05$. Sperm amounts are scaled units.",
    )
    panel(a, "A  Fertilisation outcomes before fitness weighting")
    positions = [0, 1, 3, 4]
    p_c, p_h, p_u = [], [], []
    rows = []
    for age in (0, 6):
        old = float(remaining(x0, age, h))
        for action in ("Reject", "Accept"):
            added = 0.0 if action == "Reject" else y
            denominator = 1 + old + added
            probs = (old / denominator, added / denominator, 1 / denominator)
            p_c.append(probs[0]); p_h.append(probs[1]); p_u.append(probs[2])
            rows.append((age, action, old, added, *probs))
    a.bar(positions, p_c, color=BLUE, width=0.68, label="Conspecific")
    a.bar(positions, p_h, bottom=p_c, color=ORANGE, width=0.68, label="Hybrid")
    a.bar(positions, p_u, bottom=np.array(p_c)+p_h, color=PALE_GREY, width=0.68, label="Unfertilised")
    a.set(ylim=(0, 1.23), xlim=(-0.65, 4.65), ylabel="Probability per potential egg")
    a.set_xticks(positions, ["Reject\nDay 0", "Accept\nDay 0", "Reject\nDay 6", "Accept\nDay 6"])
    a.set_yticks([0, .25, .5, .75, 1])
    a.legend(loc="upper center", ncol=3, fontsize=8.5, handlelength=1.2,
             columnspacing=0.8, frameon=False)
    csv_output("figure_2A_fertilisation_outcomes.csv", ["age_days", "action", "old_store", "new_store",
        "probability_conspecific_fertilisation", "probability_hybrid_fertilisation", "probability_unfertilised"], rows)

    panel(b, "B  Net payoff after offspring fitness and cost")
    b.fill_between(t, reject, accept, where=reject >= accept, color=BLUE, alpha=0.10, interpolate=True)
    b.fill_between(t, reject, accept, where=accept >= reject, color=ORANGE, alpha=0.12, interpolate=True)
    b.plot(t, reject, color=BLUE)
    b.plot(t, accept, color=ORANGE)
    b.axvline(switch, color="#9DA7AC", ls=":", lw=1.1)
    value = float(fertility(remaining(x0, switch, h)))
    b.plot(switch, value, "o", color=INK, markersize=4.5)
    b.annotate(f"Equal payoff\n{switch:.2f} days", xy=(switch, value), xytext=(6.15, .66),
               arrowprops={"arrowstyle": "-", "lw": 1, "color": "#65747C"}, fontsize=10)
    b.text(1.65, .91, "Reject has higher payoff", color=BLUE, ha="center", fontsize=9.5)
    b.text(7.5, .91, "Accept has higher payoff", color=ORANGE, ha="center", fontsize=9.5)
    b.text(9.85, accept[-1]+.026, "Accept", color=ORANGE, ha="right", fontsize=10.5)
    b.text(9.85, reject[-1]+.072, "Reject", color=BLUE, ha="right", fontsize=10.5)
    b.set(xlim=(0, 10), ylim=(0, 1), xlabel="Time since first mating (days)", ylabel="Net payoff per potential egg")
    b.set_xticks([0, 2, 4, 6, 8, 10])
    csv_output("figure_2B_payoffs.csv", ["age_days", "functional_conspecific_store", "payoff_reject", "payoff_accept", "accept_minus_reject"],
               zip(t, x, reject, accept, accept-reject))
    stem = "Figure_2_Remating_Tradeoff"
    save_figure(fig, stem)
    CAPTIONS.append((stem,
        "Argument: fertilisation assurance and offspring fitness can favour different actions.",
        "Panel A partitions potential eggs into conspecific fertilisations, hybrid fertilisations and fertilisation failures. At either age, accepting the new ejaculate reduces fertilisation failure but also reduces the probability of conspecific fertilisation. Panel B weights conspecific offspring by 1 and hybrid offspring by 0.5, then subtracts a mating cost of 0.05 from acceptance. Rejection gives the higher payoff early; acceptance gives the higher payoff after 4.84 days. The original conspecific sperm remain available after remating. The change in the preferred action is caused by declining conspecific availability, not a change in hybrid fitness.",
        "Scope: payoffs describe the next short reproductive bout, with a fixed pool of potential eggs and no further mating opportunity. Cost is an absolute payoff deduction, not a 5% survival reduction. The comparison assumes access to the relevant state information; it does not demonstrate that females assess sperm condition."))
    return switch


def figure_3():
    t = np.linspace(0, 10, 501)
    x = remaining(4.0, t, 2.0)
    fig, (a, b) = new_figure(
        "3  Hybrid fitness and sperm longevity shift the acceptance threshold",
        "Acceptance has the higher payoff above the boundary. Longer persistence delays the cost-free threshold.",
        r"$w_{\rm crit}(t)=\frac{x(t)}{1+x(t)}+\frac{c\,[1+x(t)+y]}{y}$",
        r"A: $x_0=4$, $h_C=2$ days, $y=4$. B: $x_0=4$, $c=0$, $y>0$. Values are illustrative.",
    )
    panel(a, "A  The boundary shifts when mating is costly")
    boundary = acceptance_boundary(x, 4.0, .05)
    free = acceptance_boundary(x, 4.0, 0.0)
    a.fill_between(t, boundary, 1, color=ORANGE, alpha=.12)
    a.fill_between(t, 0, boundary, color=BLUE, alpha=.10)
    a.plot(t, boundary, color=INK, lw=2, label="Mating cost = 0.05")
    a.plot(t, free, color="#73838C", lw=1.8, ls="--", label="No mating cost")
    a.text(7.4, .56, "Accept", color=ORANGE, fontsize=14, fontweight="semibold", ha="center")
    a.text(1.8, .18, "Reject", color=BLUE, fontsize=14, fontweight="semibold", ha="center")
    a.legend(frameon=False, fontsize=9, loc="upper right")
    a.set(xlim=(0, 10), ylim=(0, 1), xlabel="Time since first mating (days)", ylabel=r"Relative hybrid offspring fitness, $w$")
    a.set_xticks([0, 2, 4, 6, 8, 10])
    csv_output("figure_3A_acceptance_boundary.csv", ["age_days", "functional_conspecific_store", "critical_w_cost_0", "critical_w_cost_0_05"],
               zip(t, x, free, boundary))

    panel(b, "B  Longer half-life delays the cost-free threshold")
    w = np.linspace(.1, .95, 426)
    rows = []
    for h, col, style in zip([1, 2, 4], HALF_COLOURS, HALF_STYLES):
        earliest = cost_free_switch(4, h, w)
        b.plot(w, earliest, color=col, ls=style, label=f"Half-life: {h} day" + ("s" if h > 1 else ""))
        b.plot(.5, cost_free_switch(4, h, .5), "o", color=col, markerfacecolor="white", markersize=5.5, markeredgewidth=1.5)
        rows.extend(zip(w, np.full(w.size, h), earliest))
    b.set(xlim=(.1, .95), ylim=(0, 22), xlabel=r"Relative hybrid offspring fitness, $w$", ylabel="Earliest acceptance threshold (days)")
    b.set_xticks([.2, .4, .6, .8])
    b.set_yticks([0, 5, 10, 15, 20])
    b.legend(frameon=False, fontsize=9.5, loc="upper right")
    b.text(.93, 11.2, r"$t^*=\max\!\left\{0,\;h_C\log_2\!\left[\frac{x_0(1-w)}{w}\right]\right\}$",
           fontsize=11.5, ha="right")
    csv_output("figure_3B_cost_free_thresholds.csv", ["hybrid_fitness", "conspecific_half_life_days", "earliest_threshold_days"], rows)
    stem = "Figure_3_Acceptance_Thresholds"
    save_figure(fig, stem)
    CAPTIONS.append((stem,
        "Argument: lower hybrid fitness and longer sperm persistence delay beneficial heterospecific mating.",
        "Panel A rearranges the payoff inequality into a critical hybrid fitness. The solid boundary includes mating cost 0.05; the dashed boundary sets the cost to zero. Shaded regions refer to the solid boundary. At a fixed age, acceptance has the higher payoff above that boundary. Lower hybrid fitness requires a more depleted conspecific store before acceptance becomes beneficial. Panel B isolates the effect of conspecific half-life when mating is cost-free. Open circles show thresholds at hybrid fitness 0.5. The formula applies for 0 < w < 1 and a positive new ejaculate; negative calculated times are set to zero.",
        "Inference: divergence predicts a later acceptance threshold only if it lowers hybrid fitness, holding the other model components fixed. The horizontal variable in B is hybrid fitness, not divergence time. Initial store and longevity have no specified evolutionary costs here, so this figure does not establish their evolutionary optima."))


def figure_4():
    u = np.linspace(0, 8, 401)
    h_c = 2.0
    c_store = remaining(1.0, u, h_c)
    fig, (a, b) = new_figure(
        "4  Conspecific share can rise while total fertilisation falls",
        "Both ejaculates start with equal functional availability. Conspecific half-life is fixed at 2 days.",
        r"$X(u)=2^{-u/h_C},\quad Y(u)=2^{-u/h_H}$     ;     $P_C=\frac{X}{X+Y},\quad F=\frac{X+Y}{1+X+Y}$",
        "Time zero is after both ejaculates are present. Matching colours and line styles identify the same scenario in both panels.",
    )
    panel(a, "A  Conspecific share among fertilised eggs")
    panel(b, "B  Total fertilisation probability")
    rows = []
    for h_h, col, style in [(1, ORANGE, "-"), (2, GREY, "--"), (4, PURPLE, "-.")]:
        h_store = remaining(1.0, u, h_h)
        share = c_store / (c_store + h_store)
        total = fertility(c_store + h_store)
        label = f"H half-life: {h_h} day" + ("s" if h_h > 1 else "")
        a.plot(u, share, color=col, ls=style, label=label)
        b.plot(u, total, color=col, ls=style)
        rows.extend(zip(u, np.full(u.size, h_h), c_store, h_store, share, total,
                        c_store / (1+c_store+h_store), h_store / (1+c_store+h_store)))
    a.set(xlim=(0, 8), ylim=(0, 1.04), xlabel="Time since both ejaculates entered the comparison (days)",
          ylabel="Conspecific fraction of fertilisations")
    # Short two-line axis labels are easier to read in a manuscript column.
    a.set_xlabel("Time after both ejaculates are present (days)", fontsize=10.3)
    a.set_xticks([0, 2, 4, 6, 8]); a.set_yticks([0, .25, .5, .75, 1])
    a.legend(frameon=False, fontsize=9.5, loc="upper left")
    b.set(xlim=(0, 8), ylim=(0, .75), xlabel="Time after both ejaculates are present (days)",
          ylabel="Fraction of potential eggs fertilised")
    b.set_xlabel("Time after both ejaculates are present (days)", fontsize=10.3)
    b.set_xticks([0, 2, 4, 6, 8]); b.set_yticks([0, .25, .5, .75])
    b.text(7.85, .69, "All three scenarios lose\nfertilisation capacity", ha="right", va="top", fontsize=11, color="#536168")
    csv_output("figure_4_persistence_and_fertilisation.csv", ["age_after_both_ejaculates_days", "heterospecific_half_life_days",
        "conspecific_availability", "heterospecific_availability", "conspecific_share_among_fertilised",
        "probability_any_fertilisation", "probability_conspecific_fertilisation", "probability_hybrid_fertilisation"], rows)
    stem = "Figure_4_CGP_and_Total_Fertilisation"
    save_figure(fig, stem)
    CAPTIONS.append((stem,
        "Argument: a rising conspecific fertilisation share does not imply increasing reproductive output.",
        "Panel A shows conspecific share conditional on an egg being fertilised. The share increases when heterospecific sperm lose function faster, remains constant when half-lives are equal, and decreases when heterospecific sperm persist longer. Panel B shows the probability that a potential egg is fertilised at all. That probability declines in every scenario because both functional stores decline. For the one-day heterospecific half-life, conspecific share reaches about 94% at day eight, while total fertilisation falls to about 6%. Both quantities are required to interpret the biological consequence of a paternity bias.",
        "Scope: equal initial stores isolate persistence and are a separate example from Figures 2 and 3. Half-lives are fixed; there is no new input, explicit sperm use or induced response to a competitor. These are snapshots across storage ages, not cumulative egg production. Differential persistence alone can generate this bias; it does not establish a competition-specific mechanism."))


def verify_model(switch):
    """Checks target actual risks: probability sums, algebra and switch locations."""
    x = np.geomspace(1e-8, 1e4, 200)
    for y in (.1, 1, 4, 10):
        for w in (0, .2, .5, 1):
            for c in (0, .05):
                r, a = payoffs(x, y, w, c)
                decomposed = y/(1+x+y) * (w - fertility(x)) - c
                np.testing.assert_allclose(a-r, decomposed, atol=2e-15, rtol=1e-11)
                np.testing.assert_allclose(x/(1+x+y)+y/(1+x+y)+1/(1+x+y), 1)
                np.testing.assert_array_equal(a > r + 1e-10, w > acceptance_boundary(x, y, c) + 1e-9)
    # The cost-free analytical switch must give equal payoffs when positive.
    for x0 in (2, 4, 8):
        for half_life in (1, 2, 4):
            for w in (.2, .5):
                t = float(cost_free_switch(x0, half_life, w))
                if t > 0:
                    r, a = payoffs(remaining(x0, t, half_life), 4, w, 0)
                    assert abs(float(a-r)) < 1e-12
    assert abs(switch - 4.835141672526998) < 1e-10
    diagnostics = {
        "status": "passed",
        "figure_2_switch_days": switch,
        "cost_free_switch_same_parameters_days": float(cost_free_switch(4, 2, .5)),
        "figure_4_day_8_hH_1_conspecific_share": float(remaining(1,8,2)/(remaining(1,8,2)+remaining(1,8,1))),
        "figure_4_day_8_hH_1_total_fertilisation": float(fertility(remaining(1,8,2)+remaining(1,8,1))),
        "checks": ["payoff decomposition", "fertilisation probabilities sum to one", "acceptance boundary sign", "analytical switch", "numerical switch"],
    }
    (ROOT / "model_checks.json").write_text(json.dumps(diagnostics, indent=2)+"\n", encoding="utf-8")


def build_guide():
    """Keep figure artwork vector-based while adding prose with ReportLab."""
    font_dir = Path(matplotlib.get_data_path()) / "fonts/ttf"
    pdfmetrics.registerFont(TTFont("GuideSans", str(font_dir / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("GuideSansBold", str(font_dir / "DejaVuSans-Bold.ttf")))
    normal = ParagraphStyle("body", fontName="GuideSans", fontSize=10.1, leading=14.4, textColor=INK)
    bold = ParagraphStyle("argument", fontName="GuideSansBold", fontSize=11.3, leading=15, textColor=INK)
    small = ParagraphStyle("scope", fontName="GuideSans", fontSize=9.2, leading=13.2, textColor="#52616A")
    # A generous landscape page makes the two-panel figures legible and leaves
    # space for the scientific argument beneath each one.
    width, height = 936.0, 720.0
    writer = PdfWriter()
    for index, (stem, argument, caption, scope) in enumerate(CAPTIONS, start=1):
        buffer = BytesIO()
        cv = canvas.Canvas(buffer, pagesize=(width, height))
        cv.setTitle("CGP model: figures and mathematical arguments")
        cv.setAuthor("Prepared for Daniel Ortiz-Barrientos")
        left, available = 44.0, width-88.0
        y_top = 226.0
        for text, style in [(argument, bold), (caption, normal), (scope, small)]:
            para = Paragraph(text, style)
            _, para_height = para.wrap(available, y_top)
            para.drawOn(cv, left, y_top-para_height)
            y_top -= para_height+10
        if y_top < 40:
            raise RuntimeError(f"Caption overflow on page {index}: final y={y_top}")
        cv.setStrokeColorRGB(.85, .88, .89)
        cv.line(left, 36, width-left, 36)
        cv.setFont("GuideSans", 8)
        cv.setFillColorRGB(.35,.40,.44)
        cv.drawString(left, 21, "Illustrative model predictions; no empirical data fitted. Equations and assumptions: README.txt in the figure package.")
        cv.drawRightString(width-left, 21, f"{index} / {len(CAPTIONS)}")
        cv.showPage(); cv.save()
        base = PdfReader(BytesIO(buffer.getvalue())).pages[0]
        artwork = PdfReader(FIGURES / f"{stem}.pdf").pages[0]
        scale = (width-48.0) / float(artwork.mediabox.width)
        chart_height = float(artwork.mediabox.height) * scale
        # Figure top aligns at 705 points; its bottom is above the captions.
        base.merge_transformed_page(artwork, Transformation().scale(scale).translate(24, height-15-chart_height))
        writer.add_page(base)
    writer.add_metadata({"/Title": "CGP model: figures and mathematical arguments", "/Author": "Prepared for Daniel Ortiz-Barrientos"})
    with (ROOT / "CGP_Model_Figures_and_Arguments.pdf").open("wb") as stream:
        writer.write(stream)


def write_readme():
    content = """CGP MODEL FIGURES AND MATHEMATICAL ARGUMENTS
Prepared for Daniel Ortiz-Barrientos, 20 October 2026.

Purpose
-------
Four figures explain the structure of a deliberately simple model. All
numerical choices are illustrative. No values were estimated from data.

Files
-----
CGP_Model_Figures_and_Arguments.pdf: four-page guide with vector artwork,
    captions, mathematical interpretations and limits.
figures/: each figure as 300-dpi PNG and vector PDF/SVG. SVG retains editable
    text. A vector editor may require DejaVu Sans for exact font matching.
data/: numerical values used in each panel, as CSV files.
src/make_figures.py: annotated Python source that regenerates all outputs.
model_checks.json: numerical consistency checks.

Reproduce
---------
Install numpy, scipy, matplotlib, reportlab and pypdf, then run:
    python3 src/make_figures.py
Run from any working directory. Paths are resolved relative to the script.

Variables and units
-------------------
x0: initial functional conspecific store, in scaled availability units.
x(t): functional conspecific availability at storage age t.
y: functional heterospecific availability delivered at the second mating.
h_C, h_H: half-lives of functional availability, in days in these examples.
w: expected fitness contribution per hybrid offspring relative to one
   conspecific offspring; restricted here to [0,1]. It can incorporate
   survival and subsequent reproduction, but must use the same definition
   in both offspring classes.
c: extra mating cost, expressed as an absolute fitness deduction per
   potential egg over the evaluated bout. It is not a survival percentage.

Model and derivation
--------------------
Functional availability: x(t) = x0 * 2^(-t/h_C).
Assumed probability of fertilisation with availability z: F(z)=z/(1+z).
Thus z=1 is the availability giving 50% fertilisation, not one sperm.
Conditional on fertilisation, conspecific share=x/(x+y), hybrid share=y/(x+y).
The resulting unconditional probabilities are x/(1+x+y), y/(1+x+y),
and 1/(1+x+y) for conspecific, hybrid and unfertilised outcomes.

W_reject = x/(1+x).
W_accept = (x+w*y)/(1+x+y) - c.

Subtract the two expressions using the common denominator
(1+x+y)*(1+x). The numerator is:
(x+w*y)*(1+x) - x*(1+x+y) = y*[w*(1+x)-x].
Therefore:
Delta W = y/(1+x+y) * [w-x/(1+x)] - c.

For y>0, acceptance has the higher payoff exactly when:
w > w_crit(t) = x(t)/(1+x(t)) + c*(1+x(t)+y)/y.
This identity, rather than a simulation of behaviour, defines Figure 3A.

When c=0 and 0<w<1, the inequality is equivalent to x < w/(1-w).
Substitution of x=x0*2^(-t/h_C) and taking base-2 logarithms gives:
t > h_C * log2[x0*(1-w)/w].
The earliest nonnegative threshold is the maximum of this expression and
zero. At equality the two actions have equal payoff. At w=0 and c>=0,
acceptance cannot improve the payoff; at w=1 and c=0, any y>0 improves
fertilisation whenever the original store is finite. With positive cost,
acceptance never becomes strictly beneficial if c >= w*y/(1+y).

For Figure 4, time u begins when both ejaculates are present. Starting
availability is X(0)=Y(0)=1, h_C=2, and h_H is 1, 2 or 4 days.
X(u)=2^(-u/h_C), Y(u)=2^(-u/h_H).
P_C(u)=X/(X+Y), while total fertilisation is F(u)=(X+Y)/(1+X+Y).
The odds P_C/(1-P_C)=X/Y are proportional to
2^[u*(1/h_H-1/h_C)]. Consequently conspecific share increases when h_H<h_C.
The total store decreases in every scenario, so total fertilisation falls.
These are snapshots at different ages, not accumulated egg production.

Assumptions and limits
----------------------
1. One recipient background; a fixed set of potential eggs in a short bout.
2. Old sperm remain available after a new ejaculate arrives. No complete
   replacement is imposed and there is no explicit storage capacity limit.
3. Constant proportional loss of functional availability; no explicit sperm
   use, ejection, replenishment or changes induced by a competitor.
4. The saturating fertility curve and proportional sperm use are assumptions.
   The exact numerical thresholds depend on these functions.
5. Offspring fitness affects the recipient payoff, not paternity directly.
6. The payoff comparison does not establish that recipients perceive their
   state or implement the favoured action. There is no future mating option
   within the evaluated bout and no dynamic optimisation of waiting.
7. There are no specified genetic effects or costs of production, longevity
   or discrimination. These figures identify conditional fitness advantages;
   they do not establish evolutionary optima or changes in allele frequency.
8. Divergence matters here only through a separately justified effect on w
   or another parameter. No monotonic divergence relationship is assumed.
9. Differential persistence can generate conspecific bias without an induced
   competitive interaction. Single-donor controls are needed to distinguish
   baseline incompatibility from an effect of another ejaculate.
10. A pollen extension needs explicit deposition, germination and ovule
    availability. These figures depict the sperm-storage model.

Relation to established work
---------------------------
Alonzo SH and Pizzari T (2013). Selection on female remating interval is
influenced by male sperm competition strategies and ejaculate characteristics.
Philosophical Transactions of the Royal Society B 368:20120044.
https://doi.org/10.1098/rstb.2012.0044
This provides theoretical context for fertility, depletion, costs and remating.

Matute DR (2010). Reinforcement of Gametic Isolation in Drosophila.
PLoS Biology 8:e1000341. https://doi.org/10.1371/journal.pbio.1000341
This provides empirical context linking changes in the use or persistence
of heterospecific sperm with remating benefits; it did not isolate mortality
from depletion or reduced retention.

The equations and visualisations here implement the toy model developed in
this conversation. They are not figures or fitted results from those studies.

Figure captions
---------------
"""
    for stem, argument, caption, scope in CAPTIONS:
        content += "\n" + stem + "\n" + argument + "\n"
        content += textwrap.fill(caption, 78) + "\n\n" + textwrap.fill(scope, 78) + "\n"
    (ROOT / "README.txt").write_text(content, encoding="utf-8")
    (ROOT / "requirements.txt").write_text("numpy\nscipy\nmatplotlib\nreportlab\npypdf\n", encoding="utf-8")


def build_archive():
    archive = ROOT.parent / "CGP_Model_Figure_Package.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(ROOT.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                bundle.write(path, path.relative_to(ROOT.parent))
    return archive


def main():
    figure_1()
    switch = figure_2()
    figure_3()
    figure_4()
    verify_model(switch)
    write_readme()
    build_guide()
    archive = build_archive()
    print(json.dumps({"guide": str(ROOT / "CGP_Model_Figures_and_Arguments.pdf"),
                      "archive": str(archive), "figures": len(CAPTIONS),
                      "switch_days": switch, "checks": "passed"}, indent=2))


if __name__ == "__main__":
    main()
