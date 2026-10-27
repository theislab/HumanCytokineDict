"""Explain how one entry in ``plot_secondary_overview`` is constructed.

This module deliberately lives outside the analysis notebook.  It consumes the
objects already produced by the secondary-response analysis and displays
the cell-, lineage-, pair-, and causality-level decisions for one
``(query_cyto, secondary_cyto)`` pair.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import fill
from typing import Callable

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors as mcolors
from matplotlib import patheffects
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns


def _bool_word(value) -> str:
    if pd.isna(value):
        return "unknown"
    return "yes" if bool(value) else "no"


def _fmt(value, digits=2) -> str:
    return "NA" if pd.isna(value) else f"{float(value):.{digits}f}"


def _single_pair_row(pair_table, query_cyto, secondary_cyto) -> pd.Series:
    selected = pair_table.loc[
        pair_table["query_cyto"].eq(query_cyto)
        & pair_table["secondary"].eq(secondary_cyto)
    ]
    if selected.empty:
        raise KeyError(f"No secondary-response pair found for {(query_cyto, secondary_cyto)!r}.")
    if len(selected) > 1:
        raise ValueError(f"Expected one row for {(query_cyto, secondary_cyto)!r}; found {len(selected)}.")
    return selected.iloc[0]


def audit_secondary_overview_entry(query_cyto, secondary_cyto, *, pair_table, pair_details_by_pair, producer_details_by_pair, config, strong_impact_by_cell_type):
    """Assemble saved analysis outputs for display; no analytical results are recomputed."""
    row = _single_pair_row(pair_table, query_cyto, secondary_cyto)
    key = (query_cyto, secondary_cyto)
    details = pair_details_by_pair[key]
    producer_details = producer_details_by_pair[key]
    evidence = dict(details)
    evidence["lineage_table"] = pd.DataFrame({"evidence": details["lineage_evidence"], "contradiction": details["lineage_contradiction"], "supported": details["lineage_support"]})
    for name in ("pair_evidence", "pair_contradiction", "eligible_lineages", "supporting_lineages", "confidence"):
        evidence[name] = row[name]

    all_donors = pd.Index(producer_details["treated_donors"]).union(pd.Index(producer_details["baseline_donors"]), sort=False)
    donor_table = pd.DataFrame(index=all_donors)
    donor_table["paired"] = all_donors.isin(producer_details["treated_donors"]) & all_donors.isin(producer_details["baseline_donors"])
    donor_table["retained"] = all_donors.isin(producer_details["responsive_donors"])
    donor_table["signed_log2fc"] = producer_details["signed_log2fc"]
    donor_table["drop_reason"] = "retained"
    donor_table.loc[~donor_table["paired"], "drop_reason"] = "unpaired donor"
    donor_table.loc[donor_table["paired"] & ~donor_table["retained"], "drop_reason"] = "not responsive"
    share_matrix = pd.DataFrame(index=all_donors, columns=producer_details["donor_shares_all"].columns, dtype=float)
    share_matrix.loc[producer_details["donor_shares_all"].index] = producer_details["donor_shares_all"]
    producer = {"donor_table": donor_table, "directional_shares_all": share_matrix, "receptor_class": producer_details["producer_receptor_classes"]}
    direction_sign = {"secondary increased": 1, "secondary reduced": -1}[row["direction"]]
    anchors = list(config.anchor_cell_types)
    return {
        "key": key,
        "row": row,
        "similarity": details["similarity"],
        "secondary_receptor": details["secondary_receptor"],
        "query_receptor": details["query_receptor"],
        "query_anchor_strong": strong_impact_by_cell_type.loc[query_cyto, anchors].any(),
        "secondary_anchor_strong": strong_impact_by_cell_type.loc[secondary_cyto, anchors].any(),
        "evidence": evidence,
        "magnitude_score": row["magnitude_score"],
        "priority": row["priority"],
        "signed_priority": direction_sign * row["priority"],
        "supported_cells": row["supported_recipient_cells"],
        "receptor_low_recipients": row["query_receptor_low_recipient_cells"],
        "secondary_required": row["secondary_required"],
        "direct_producer_share": row["direct_producer_share"],
        "indirect_producer_share": row["indirect_producer_share"],
        "best_intermediate": row["best_intermediate"],
        "best_intermediate_overlap": row["best_intermediate_producer_overlap"],
        "best_intermediate_common_donors": row["best_intermediate_common_donors"],
        "causal_class": row["causal_class"],
        "producer_audit": producer,
        "producer_details": producer_details,
    }


def _decision_explanation(audit, config):
    """Show each ordered classifier condition using its saved operands."""
    d, i = audit["direct_producer_share"], audit["indirect_producer_share"]
    dt, it = config.direct_share_min, config.indirect_share_min
    pct = lambda x: "NA" if pd.isna(x) else f"{x:.1%}"
    required = audit["secondary_required"]
    route = pd.notna(audit["best_intermediate"])
    conditions = [
        ("Not evaluable", "priority missing", pd.isna(audit["priority"])),
        ("Unsupported similarity", f"supported recipients = {len(audit['supported_cells'])}; need 0", not audit["supported_cells"]),
        ("Receptor unresolved", "input receptor coverage missing", pd.isna(required)),
        ("Possibly incidental", f"input-receptor-low recipients = {len(audit['receptor_low_recipients'])}; need 0", pd.notna(required) and not required),
        ("Shared direct/indirect", f"D {pct(d)} <= {pct(dt)} AND I {pct(i)} <= {pct(it)}", d <= dt and i <= it),
        ("Immediate secondary", f"D {pct(d)} > {pct(dt)}", d > dt),
        ("Probable tertiary", f"I {pct(i)} > {pct(it)} AND accepted route: {'yes' if route else 'NO'}", i > it and route),
        ("Unresolved indirect", "none of the preceding conditions matched", True),
    ]
    rows, selected = [], None
    for idx, (label, measurement, matched) in enumerate(conditions):
        status = "not reached" if selected is not None else "SELECTED" if matched else "no"
        rows.append([label, measurement, status])
        if selected is None and matched:
            selected = idx
    if audit["causal_class"] == "unresolved indirect":
        if pd.isna(d) or pd.isna(i):
            reason = "Producer attribution is unavailable; a production route cannot be assigned."
        elif i > it and not route:
            reason = f"Indirect production ({pct(i)}) clears {pct(it)}, but no intermediate passed the route checks below."
        else:
            reason = "Neither a shared/direct attribution nor a qualifying tertiary route was established."
    else:
        reason = conditions[selected][1]
    expected = ["not evaluable", "unsupported similarity", "unresolved receptor coverage",
                "possibly incidental similarity", "shared direct/indirect", "immediate secondary",
                "probable tertiary", "unresolved indirect"][selected]
    if expected != audit["causal_class"]:
        reason = "STALE RESULT: stored label differs from these checks. Rerun classifier with this config. " + reason
    return rows, selected, reason


def _route_display_rows(row, config, formatter, pair_table):
    records = row.get("route_search_audit")
    if not isinstance(records, list):
        return [], "Route rejection details not saved. Rerun add_secondary_path_classes, then this figure."
    if not records:
        return [], (f"Search not entered: requires a receptor-low supported recipient and indirect share > {config.indirect_share_min:.1%}.")
    eligible = pair_table.loc[
        pair_table["query_cyto"].eq(row["query_cyto"])
        & pair_table["confidence"].isin(config.consistent_confidence)
        & pair_table["medium_log2fc"].abs().gt(config.supernatant_min_log2fc),
        "secondary",
    ]
    records = [record for record in records if record["intermediate"] in set(eligible)]
    if not records:
        return [], "No intermediates with consistent input-response support and sufficient supernatant change."
    # Focus the display only; preserve the complete saved search audit.
    # Group identical failed gates among the eligible intermediates.
    groups = {}
    for r in records:
        key = (r["first_edge"], r["second_edge"], r["result"])
        groups.setdefault(key, []).append(formatter(r["intermediate"]))
    rows = []
    for (first, second, result), names in groups.items():
        rows.append([fill(", ".join(names), 27, break_long_words=False), fill(first, 34, break_long_words=False), fill(second, 34, break_long_words=False), fill(result, 36, break_long_words=False)])
    accepted = sum(r["result"].startswith("accepted") for r in records)
    both = sum(r["first_edge"] == r["second_edge"] == "pass" for r in records)
    return rows, f"{len(records)} supported input-response candidates -> {both} pass both edge gates -> {accepted} accepted routes."


def plot_secondary_overview_explainer(query_cyto, secondary_cyto, *, pair_table, pair_details_by_pair, producer_details_by_pair, config,
                                      strong_impact_by_cell_type, label_formatter: Callable[[str], str] | None = None, figsize_cm=None, savepath=None):
    """Plot saved pair and producer details, returning (fig, display_details).

    Pass the config used for the calculation. Strong-impact input supplies the anchor summary and the input-cytokine
    context row; that context row does not change the recipient-support gates.
    Unpaired donors have no calculated fold change or producer shares.
    """
    audit = audit_secondary_overview_entry(query_cyto, secondary_cyto, pair_table=pair_table, pair_details_by_pair=pair_details_by_pair,
                                           producer_details_by_pair=producer_details_by_pair, config=config, strong_impact_by_cell_type=strong_impact_by_cell_type)

    receptor_threshold = config.receptor_min
    similarity_threshold = config.similarity_min
    supernatant_threshold = config.supernatant_min_log2fc
    magnitude_saturation = config.magnitude_saturation
    min_change_per_donor = config.supernatant_min_log2fc_per_donor
    direct_share_threshold = config.direct_share_min
    indirect_share_threshold = config.indirect_share_min
    route_producer_overlap_threshold = config.route_overlap_min
    min_route_common_donors = config.route_min_common_donors

    fmt_label = (lambda value: value) if label_formatter is None else label_formatter
    row = audit["row"]
    evidence = audit["evidence"]
    lineage_table = evidence["lineage_table"]
    producer = audit["producer_audit"]
    donor_table = producer["donor_table"]
    cells = list(audit["similarity"].index)

    # --- Layout: figure size and panel positions (centimetres) ---
    producer_height = max(1.4, 0.32 * len(donor_table))
    route_rows, route_summary_text = _route_display_rows(row, config, fmt_label, pair_table)
    route_row_heights = [max(value.count("\n") + 1 for value in r) * 0.26 + 0.18 for r in route_rows]
    route_height = max(0.65, sum(route_row_heights) + 0.4)
    natural_height = 15.9 + producer_height + route_height
    if figsize_cm is None:
        figsize_cm = (18.0, natural_height)
    if len(figsize_cm) != 2 or not 0 < figsize_cm[0] <= 18 or figsize_cm[1] <= 0:
        raise ValueError("figsize_cm must be positive (width, height), with width <= 18 cm.")
    audit["figure_size_cm"] = tuple(float(value) for value in figsize_cm)

    fig = plt.figure(figsize=tuple(value / 2.54 for value in figsize_cm))
    def panel(x, top, width, height):
        return fig.add_axes([x / 18, 1 - (top + height) / natural_height, width / 18, height / natural_height])
    
    overall_title = fig.suptitle(f"{fmt_label(query_cyto)} to {fmt_label(secondary_cyto)} secondary effects:", 
                                x=0.6 / 18, y=1 - 0.18 / natural_height, ha="left", fontweight="bold")
    ax_filter = panel(5.8, 0.10, 9.5, 0.55)
    evidence_top = 1.55
    ax_cells = panel(3.4, evidence_top, 10.2, 2.85)
    ax_lineage = panel(14.1, evidence_top, 3.25, 1.45)
    producer_top = 6.15
    ax_donors = panel(1.3, producer_top, 1.65, producer_height)
    ax_producers = panel(3.4, producer_top, 10.2, producer_height)
    ax_totals = panel(14.1, producer_top, 2.0, producer_height)
    bottom_top = producer_top + producer_height + 1.65
    ax_verdict = panel(0.6, bottom_top, 12.5, 4.0)
    ax_result = panel(14.1, bottom_top, 3.0, 4.0)
    ax_routes = panel(0.6, bottom_top + 6.1, 16.2, route_height)
    colorbar_ax = panel(16.6, producer_top, 0.12, producer_height)

    # Wrap cell names instead of allocating tall rows to rotated labels.
    cell_labels = [fill(str(cell).replace("CD56low", "CD56 low").replace("CD56hi", "CD56 hi"), width=5 if len(cells) > 8 else 12,
                        break_long_words=False, break_on_hyphens=False) for cell in cells]
    all_axes = [ax_filter, ax_cells, ax_lineage, ax_donors, ax_producers, ax_verdict, ax_result, ax_totals, ax_routes]
    for ax in all_axes:
        ax.set_facecolor("#F7F7F7")

    # 1. Pair-level prefilter, drawn as a literal left-to-right pipeline.
    ax_filter.axis("off")
    candidate_heading = ax_filter.text(0, 0.43, "1  Candidate check", transform=ax_filter.transAxes, ha="left", va="center", fontsize=6, fontweight=900)
    candidate_heading.set_path_effects([patheffects.withStroke(linewidth=0.30, foreground=candidate_heading.get_color())])
    pair_checks = [
        (f"Supernatant change\n{abs(row['medium_log2fc']):.2f} > {supernatant_threshold:g}: PASS", "#D9EEE9"),
        (f"anchor-cell impact\n{fmt_label(query_cyto)} {_bool_word(audit['query_anchor_strong'])}; {fmt_label(secondary_cyto)} {_bool_word(audit['secondary_anchor_strong'])}",
         "#D9EEE9" if audit["query_anchor_strong"] and audit["secondary_anchor_strong"] else "#F3D6D6"),
        ("candidate\nadmitted", "#CFE4CF"),
    ]
    # Heading occupies the first 22%; the three boxes have separate slots.
    x_positions, widths = [0.25, 0.56, 0.88], [0.23, 0.26, 0.12]
    for i, ((text, color), x, width) in enumerate(zip(pair_checks, x_positions, widths)):
        ax_filter.text(x + width / 2, 0.43, text, transform=ax_filter.transAxes, ha="center", va="center", fontsize=6,
                       bbox=dict(boxstyle="round,pad=0.30", facecolor=color, edgecolor="#777777"))
        if i < len(pair_checks) - 1:
            ax_filter.annotate("", xy=(x_positions[i + 1] - 0.01, 0.43), xytext=(x + width + 0.01, 0.43), xycoords=ax_filter.transAxes,
                               arrowprops=dict(arrowstyle="->", color="#555555", lw=1.1))

    # 2. All-cell gate matrix. Values are shown in the cells; colors show pass/fail.
    ax_cells.set_title("2  Recipient evidence", x=(0.6 - 3.4) / 10.2, loc="left", fontweight="bold", fontsize=6)
    secondary_receptor = audit["secondary_receptor"].loc[cells]
    secondary_impact = evidence["secondary_impact"].loc[cells]
    aligned = evidence["aligned_similarity"].loc[cells]
    eligible = evidence["eligible"].loc[cells]
    support = evidence["support"].loc[cells]
    query_receptor = audit["query_receptor"].loc[cells]

    # Input impact is context only; secondary eligibility and support use the saved gates.
    query_impact = strong_impact_by_cell_type.reindex(index=[query_cyto], columns=cells).iloc[0]
    query_classes = producer["receptor_class"].loc[cells]
    required_cells = pd.Index(cells).isin(audit["receptor_low_recipients"])
    audit["query_impact"] = query_impact
    # -2 missing, -1 skipped, 0 failed, 1 passed.
    state = np.full((8, len(cells)), -2.0)
    state[0] = np.where(query_classes.eq("missing") | query_receptor.isna(), -2, query_classes.eq("indirect"))
    state[1] = np.where(query_impact.isna(), -2, query_impact.fillna(False).astype(bool))
    state[2] = np.where(secondary_receptor.isna(), -2, secondary_receptor.ge(receptor_threshold))
    state[3] = secondary_impact.astype(int)
    state[4] = eligible.astype(int)
    state[5] = np.where(~eligible, -1, np.where(aligned.isna(), -2, aligned.ge(similarity_threshold).astype(int)))
    state[6] = np.where(~eligible | aligned.isna(), -1, support.astype(int))
    state[7] = np.where(~support, -1, np.where(query_classes.eq("missing"), -2, required_cells))
    state_cmap = mcolors.ListedColormap(["white", "#E2E2E2", "#E9B7B2", "#4B9A91"])
    state_norm = mcolors.BoundaryNorm([-2.5, -1.5, -0.5, 0.5, 1.5], 4)
    ax_cells.imshow(state, cmap=state_cmap, norm=state_norm, aspect="auto")
    row_labels = [
        f"{fmt_label(query_cyto)} receptor < {receptor_threshold:g}", f"{fmt_label(query_cyto)} strong impact (context)",
        f"{fmt_label(secondary_cyto)} receptor >= {receptor_threshold:g}", f"{fmt_label(secondary_cyto)} strong impact",
        "secondary eligible", f"aligned r >= {similarity_threshold:g}", "supported (eligible + r)", "support + input receptor low",
    ]
    annotations = [
        ["NA" if query_classes.loc[cell] == "missing" else _fmt(query_receptor.loc[cell], 1) for cell in cells],
        [_bool_word(value) for value in query_impact],
        [_fmt(value, 1) for value in secondary_receptor], ["yes" if value else "no" for value in secondary_impact],
        ["yes" if value else "no" for value in eligible],
        ["skip" if not eligible.iloc[i] else _fmt(aligned.iloc[i], 2) for i in range(len(cells))],
        ["yes" if value else "no" for value in support],
        ["skip" if not support.loc[cell] else "NA" if query_classes.loc[cell] == "missing" else "yes" if required_cells[i] else "no" for i, cell in enumerate(cells)],
    ]
    for row_i in range(state.shape[0]):
        for col_i in range(state.shape[1]):
            ax_cells.text(col_i, row_i, annotations[row_i][col_i], ha="center", va="center", fontsize=6)
    ax_cells.set_xticks(np.arange(len(cells)), cell_labels, rotation=0, ha="center", fontsize=6)
    ax_cells.set_yticks(np.arange(len(row_labels)), row_labels, fontsize=6)
    ax_cells.set_xticks(np.arange(-0.5, len(cells), 1), minor=True)
    ax_cells.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax_cells.grid(which="minor", color="white", linewidth=1)
    ax_cells.tick_params(which="both", length=0)
    ax_cells.axhline(1.5, color="#475569", linewidth=1.0)
    ax_cells.axhline(6.5, color="#475569", linewidth=1.0)

    # 3. Aggregate cell evidence within lineage, then take the mean across lineages.
    ax_lineage.set_title("3  Lineage evidence", loc="left", fontweight="bold", fontsize=6)
    if lineage_table.empty:
        ax_lineage.axis("off")
        ax_lineage.text(0.5, 0.5, "No eligible lineages", ha="center", va="center")
    else:
        positions = np.arange(len(lineage_table))
        ax_lineage.bar(positions, lineage_table["evidence"], color=["#168A8A" if value else "#BDBDBD" for value in lineage_table["supported"]])
        ax_lineage.bar(positions, -lineage_table["contradiction"], color="#D98989")
        ax_lineage.axhline(similarity_threshold, color="black", ls="--", lw=0.8)
        ax_lineage.axhline(0, color="black", lw=0.6)
        lineage_labels = ["My" if value == "myeloid" else value for value in lineage_table.index]
        ax_lineage.set_xticks(positions, lineage_labels, fontsize=6)
        ax_lineage.set_ylabel("")
        ax_lineage.set_xlabel("weighted aligned r", fontsize=6)
        ax_lineage.yaxis.tick_right()
        ax_lineage.tick_params(length=0, labelsize=6)
        ax_lineage.spines[["top", "right"]].set_visible(False)

    # Confidence, mean evidence and the overview color calculation belong to panel 3.
    lineage_summary = (f"{evidence['confidence']}\nMean evidence = {_fmt(evidence['pair_evidence'])}\n"
        f"Magnitude = {audit['magnitude_score']:.2f}\nPriority = {_fmt(evidence['pair_evidence'])} x {audit['magnitude_score']:.2f}\n"
        f"Signed color = {audit['signed_priority']:+.2f}")
    ax_lineage.text(0, -.60, lineage_summary, transform=ax_lineage.transAxes, va="top", fontsize=6, linespacing=1.3)

    # 5. Horizontal bars share the heatmap's donor positions and labels.
    ax_donors.set_title("5  Donor filter", x=(0.6 - 1.3) / 1.65, loc="left", fontweight="bold", fontsize=6)
    donor_y = np.arange(len(donor_table)) + 0.5
    donor_colors = np.where(donor_table["retained"], "#2F8F83", "#D98C83")
    paired = donor_table["paired"].to_numpy()
    ax_donors.barh(donor_y[paired], donor_table.loc[donor_table["paired"], "signed_log2fc"], color=donor_colors[paired], height=0.70)
    ax_donors.scatter(np.zeros((~paired).sum()), donor_y[~paired], marker="x", color="#555555", zorder=4)
    ax_donors.axvline(min_change_per_donor, color="black", ls="--", lw=0.9)
    ax_donors.axvline(0, color="black", lw=0.6)
    ax_donors.set_ylim(len(donor_table), 0)
    ax_donors.set_yticks(donor_y, [str(donor) for donor in donor_table.index], fontsize=6)
    ax_donors.set_xlabel(f"signed log2FC\nkeep > {min_change_per_donor:g}", fontsize=6)
    ax_donors.xaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
    ax_donors.tick_params(length=0, labelsize=6)
    ax_donors.spines[["top", "right"]].set_visible(False)

    # 6. Direction-aligned positive production shares for every donor and cell.
    producer_title = ax_producers.set_title("6a  Producer shares (%) by cell type", loc="left", fontweight="bold", fontsize=6)
    share_matrix = producer["directional_shares_all"].loc[donor_table.index, cells]
    share_percent = 100 * share_matrix
    sns.heatmap(share_percent, vmin=0, vmax=100, cmap="Blues", annot=True, fmt=".0f", annot_kws={"fontsize": 6}, cbar=True, cbar_ax=colorbar_ax,
        cbar_kws={"ticks": [0, 50, 100]}, linewidths=0.35, linecolor="white", ax=ax_producers)
    colorbar_ax.set_title("%", fontsize=6)
    receptor_class = producer["receptor_class"].loc[cells]
    direct_cells = set(receptor_class.index[receptor_class.eq("direct")])
    xlabels = [
        f"{cell_labels[i]}\n[{'D' if cell in direct_cells else 'I'}]"
        if receptor_class.loc[cell] != "missing"
        else f"{cell_labels[i]}\n[NA]"
        for i, cell in enumerate(cells)
    ]
    ax_producers.set_xticks(np.arange(len(cells)) + 0.5, xlabels, rotation=0, ha="center", fontsize=6)
    ax_producers.set_yticks([])
    ax_producers.set_xlabel("")
    ax_producers.set_ylabel("")
    ax_producers.tick_params(axis="both", which="both", length=0)
    for row_i, donor in enumerate(donor_table.index):
        if not donor_table.loc[donor, "retained"]:
            ax_producers.add_patch(plt.Rectangle((0, row_i), len(cells), 1, fill=False, edgecolor="#A33A32", linewidth=1.0, linestyle="--", clip_on=False))
    # --- 6b. Direct/indirect donor totals and medians ---
    direct_columns = producer["receptor_class"].eq("direct")
    totals = producer["directional_shares_all"].loc[:, direct_columns].sum(axis=1, min_count=1)
    if producer["receptor_class"].eq("missing").any():
        totals[:] = np.nan
    totals = totals.where(donor_table["retained"])
    audit["donor_direct_totals"] = totals
    total_frame = pd.DataFrame({"Direct": totals, "Indirect": 1 - totals})
    sns.heatmap(100 * total_frame, ax=ax_totals, cmap="Blues", vmin=0, vmax=100, annot=True, fmt=".1f", cbar=False, annot_kws={"fontsize": 6}, linewidths=.5, linecolor="white")
    ax_totals.set_title("6b  Totals (%)", loc="left", fontsize=6)
    for donor_i, donor in enumerate(total_frame.index):
        if not donor_table.loc[donor, "retained"]:
            for x in (.5, 1.5):
                ax_totals.text(x, donor_i + .5, "DROP", ha="center", va="center", fontsize=6, color="#555555")
    ax_totals.set_xticks([.5, 1.5], ["Direct", "Indirect"], rotation=0)
    ax_totals.set_yticks([])
    ax_totals.tick_params(length=0)
    ax_totals.set_xlabel(f"Median (n={int(donor_table['retained'].sum())})\nD {audit['direct_producer_share']:.1%} / I {audit['indirect_producer_share']:.1%}")
    ax_producers.set_xlabel("")

    # --- 7. Classification decision table ---
    decision_rows, selected_rule, reason = _decision_explanation(audit, config)
    audit["decision_rows"] = decision_rows
    audit["decision_reason"] = reason
    ax_verdict.set_title("7  Why this classification? Read top to bottom; stop at SELECTED", loc="left", fontsize=6)
    ax_verdict.axis("off")
    table = ax_verdict.table(cellText=decision_rows, colLabels=["Possible label", "Actual condition", "Matches?"], colWidths=[.27, .60, .13], cellLoc="left", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(6)
    for col in range(3):
        table[(0, col)].set_facecolor("#E6E6E6")
        table[(selected_rule + 1, col)].set_facecolor("#D9EEE9")
    ax_verdict.text(0, -.10, fill(reason, 105), transform=ax_verdict.transAxes, va="top", fontsize=6, fontweight="bold")
    # --- 9. Focused intermediate-route audit ---
    ax_routes.set_title("9  Intermediate-route audit: input -> M -> secondary", loc="left", fontsize=6, pad=24)
    ax_routes.axis("off")
    ax_routes.annotate(route_summary_text, xy=(0, 1), xycoords="axes fraction", xytext=(0, 4), textcoords="offset points", fontsize=6, va="bottom")
    if route_rows:
        route_table = ax_routes.table(cellText=route_rows, colLabels=["Intermediate M", "Input -> M edge", "M -> secondary edge", "Later checks / result"],
            colWidths=[.21, .26, .26, .27], cellLoc="left", bbox=[0, 0, 1, 1])
        route_table.auto_set_font_size(False)
        route_table.set_fontsize(6)
        for col in range(4):
            route_table[(0, col)].set_height(.4 / route_height)
            route_table[(0, col)].set_facecolor("#E6E6E6")
        for ri, height in enumerate(route_row_heights, 1):
            for col in range(4):
                route_table[(ri, col)].set_height(height / route_height)
    ax_routes.text(0, -.06 / route_height,
        fill(f"Shown: input -> M has consistent support and abs(supernatant log2FC) > {config.supernatant_min_log2fc:g}. Each edge requires confidence in {config.consistent_confidence}, a supported recipient, and direct producer share > {config.direct_share_min:.1%}. "
        f"Then check compatible signs, >= {config.route_min_common_donors} shared responsive donors, overlapping receptor-defined producer cells, "
        f">= {config.route_min_common_donors} donors with positive relevant production, and median producer overlap >= {config.route_overlap_min:.1%}.", 155),
        transform=ax_routes.transAxes, va="top", fontsize=6)
    # 8. Exact final visual encoding in plot_secondary_overview.
    ax_result.set_title("8  Overview entry", loc="left", fontweight="bold", fontsize=6)
    cmap = sns.color_palette("vlag", as_cmap=True)
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    square_color = cmap(norm(np.clip(audit["signed_priority"], -1, 1)))
    confidence = evidence["confidence"]
    supported_confidence = row["consistent_support"]
    ax_result.scatter([0.5], [0.70], s=1400, marker="s", facecolor=square_color, edgecolor="black" if supported_confidence else "none", linewidth=2, transform=ax_result.transAxes, clip_on=False)
    marker_style = config.causal_styles.get(audit["causal_class"])
    if marker_style is not None:
        marker = marker_style["marker"]
        white_inner = marker in config.white_inner_set
        ax_result.add_line(Line2D([0.5], [0.70], marker=marker, linestyle="", color="#333333",
            markerfacecolor="white" if white_inner else "#333333", markeredgecolor="#333333" if marker != "X" else "none",
            markeredgewidth=0.65 if marker != "X" else 0.3, markersize=11, transform=ax_result.transAxes, clip_on=False))
    marker_label = marker_style["label"] if marker_style is not None else audit["causal_class"]
    ax_result.text(0.5, 0.40, f"3 -> color: {audit['signed_priority']:+.2f}\n3 -> border: {_bool_word(supported_confidence)}\n7 -> marker:\n{fill(marker_label, 25)}",
        transform=ax_result.transAxes, ha="center", va="top", fontsize=6)
    ax_result.set_xlim(0, 1)
    ax_result.set_ylim(0, 1)
    ax_result.axis("off")

    # --- Footer, typography and export ---
    fig.text(0.035, 0.006, "Recipient matrix: teal = pass; red = fail; grey = skipped; white = missing. Donors: teal = KEEP; red / x = DROP.\nD/I: input receptor high/low. Sum D/I shares within each donor (6a -> 6b); reported medians use KEEP donors only.", color="#475569")
    # Some plotting environments resolve ``bold`` to a visually light font
    # face at 6 pt.  Reinforce only the numbered step headings so their visual
    # hierarchy remains unambiguous in the exported figure.
    for ax in all_axes:
        step_title = ax._left_title
        step_title.set_fontweight(900)
        step_title.set_path_effects([patheffects.withStroke(linewidth=0.30, foreground=step_title.get_color())])
    for text_artist in fig.findobj(match=mpl.text.Text):
        text_artist.set_fontsize(6)
    overall_title.set_fontsize(8)
    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        # Do not use bbox_inches="tight": preserving the requested physical
        # canvas guarantees that the exported width does not exceed figsize_cm.
        fig.savefig(savepath, dpi=300)
    return fig, audit
