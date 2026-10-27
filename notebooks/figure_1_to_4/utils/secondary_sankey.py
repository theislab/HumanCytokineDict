import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .plotting_utils import cm_to_px


_DIRECTION_SIGN = {
    "secondary increased": 1,
    "secondary reduced": -1,
}

_DIRECTION_COLOR = {
    "secondary increased": "rgba(190, 55, 55, 0.48)",
    "secondary reduced": "rgba(55, 95, 190, 0.48)",
}

# Matches the circle/triangle route symbols in ``plot_secondary_overview``.
_CAUSAL_SYMBOL = {
    "immediate secondary": "○",
    "probable tertiary": "△",
}


def _as_numeric_series(value):
    if value is None:
        return pd.Series(dtype=float)
    if isinstance(value, pd.Series):
        return value.astype(float)
    return pd.Series(value, dtype=float)


def _trim_and_normalize(weights, minimum_share):
    weights = _as_numeric_series(weights).replace([np.inf, -np.inf], np.nan).dropna()
    weights = weights.clip(lower=0)
    weights = weights.loc[weights.gt(0)]
    if weights.empty:
        return weights

    weights = weights / weights.sum()
    retained = weights.loc[weights.ge(minimum_share)]
    if retained.empty:
        retained = weights.nlargest(1)
    return retained / retained.sum()


def plot_secondary_causal_sankey(
    query_cyto,
    pair_table,
    producer_shares_by_pair,
    support_by_pair,
    similarity_by_pair,
    receptor_expression_by_cytokine,
    cell_abundance,
    receptor_threshold=8,
    direction=None,
    secondary=None,
    causal_classes=("immediate secondary", "probable tertiary"),
    confidence_levels=("cross-lineage support", "single-lineage supported"),
    n_pairs=8,
    minimum_priority=None,
    minimum_producer_share=0.05,
    minimum_recipient_share=0.05,
    recipient_requires_no_query_receptor=True,
    cell_type_order=None,
    figsize_cm=(18, 12),
    label_formatter=str,
    title=None,
    font_size=6,
    show_classification_symbols=True,
):
    """Plot causally resolved secondary-cytokine paths for one input cytokine.

    Immediate path
        input -> directly responsive producer -> secondary -> supported recipient

    Tertiary path
        input -> directly responsive producer -> intermediate cytokine
              -> intermediate-responsive producer -> secondary -> supported recipient

    Ribbon width is the pair-level priority from ``plot_secondary_overview``
    multiplied by the fraction of production explained by the displayed causal
    class. Within a stage, that flow is distributed according to producer shares
    or direction-aligned recipient response similarity. By default, recipient
    cells expressing the input-cytokine receptor are excluded: their response
    can be direct and is therefore not secondary-attributable.

    Returns the Plotly figure, selected pair rows, and a tidy table containing
    every displayed edge. The edge table is intended to make the causal claims
    in the figure auditable.
    """
    required_columns = {
        "query_cyto",
        "secondary",
        "direction",
        "priority",
        "confidence",
        "causal_class",
        "direct_producer_share",
        "indirect_producer_share",
        "best_intermediate",
    }
    missing = required_columns - set(pair_table.columns)
    if missing:
        raise ValueError(f"pair_table is missing columns: {sorted(missing)}")

    pairs = pair_table.loc[
        pair_table["query_cyto"].eq(query_cyto)
        & pair_table["causal_class"].isin(causal_classes)
        & pair_table["confidence"].isin(confidence_levels)
        & pair_table["priority"].notna()
    ].copy()
    if direction is not None:
        pairs = pairs.loc[pairs["direction"].eq(direction)]
    if secondary is not None:
        pairs = pairs.loc[pairs["secondary"].eq(secondary)]
    if minimum_priority is not None:
        pairs = pairs.loc[pairs["priority"].ge(minimum_priority)]
    if n_pairs is not None:
        pairs = pairs.nlargest(n_pairs, "priority")
    if pairs.empty:
        raise ValueError(f"No causally resolved secondary responses found for {query_cyto!r}.")

    pair_lookup = (
        pair_table.sort_values("priority", ascending=False)
        .drop_duplicates(["query_cyto", "secondary"])
        .set_index(["query_cyto", "secondary"])
    )
    cell_abundance = _as_numeric_series(cell_abundance).reindex(
        receptor_expression_by_cytokine.columns
    ).fillna(0)

    def receptor_values(stimulus, cells):
        if stimulus not in receptor_expression_by_cytokine.index:
            return pd.Series(np.nan, index=cells, dtype=float)
        return receptor_expression_by_cytokine.loc[stimulus].reindex(cells).astype(float)

    def producer_weights(stimulus, product, direct, allowed_cells=None):
        shares = _as_numeric_series(producer_shares_by_pair.get((stimulus, product)))
        if shares.empty:
            return shares
        receptors = receptor_values(stimulus, shares.index)
        if direct:
            shares = shares.loc[receptors.ge(receptor_threshold)]
        else:
            shares = shares.loc[receptors.notna() & receptors.lt(receptor_threshold)]
        if allowed_cells is not None:
            shares = shares.reindex(list(allowed_cells)).dropna()
        return _trim_and_normalize(shares, minimum_producer_share)

    def recipient_weights(row):
        key = (row["query_cyto"], row["secondary"])
        support = support_by_pair.get(key, pd.Series(dtype=bool))
        support = pd.Series(support, dtype="boolean").fillna(False).astype(bool)
        similarity = _as_numeric_series(similarity_by_pair.get(key))
        sign = _DIRECTION_SIGN[row["direction"]]
        aligned = (sign * similarity).clip(lower=0)
        # Recipient width should represent response similarity, not how common
        # that cell type is in PBMCs.  Cell abundance is intentionally not used
        # here, so CD16 Mono is not visually down-weighted for being rarer.
        weights = aligned.where(support, 0)
        if recipient_requires_no_query_receptor:
            query_receptor = receptor_values(query_cyto, aligned.index)
            # Require known low input-receptor expression.  Unknown expression
            # is not enough to claim that a response is secondary-mediated.
            non_direct = query_receptor.notna() & query_receptor.lt(receptor_threshold)
            weights = weights.where(non_direct, 0)
        return _trim_and_normalize(weights, minimum_recipient_share)

    node_rows = {}
    edge_rows = []

    cell_type_order = list(cell_type_order or [])
    cell_order = {cell: i for i, cell in enumerate(cell_type_order)}

    def register_node(node_id, label, stage, color, cell_type=None):
        # A cytokine can be both a displayed final secondary and an
        # intermediate for another displayed route.  Retain one node in its
        # causal position rather than duplicating it by role.
        if node_id in node_rows:
            return
        node_rows[node_id] = {
            "node_id": node_id,
            "label": label,
            "stage": stage,
            "color": color,
            "cell_type": cell_type,
        }

    def add_edge(
        source, target, value, row, segment, causal_fraction,
        intermediate=None, edge_direction=None,
    ):
        edge_rows.append({
            "source": source,
            "target": target,
            "value": float(value),
            "segment": segment,
            "query_cyto": query_cyto,
            "secondary": row["secondary"],
            "direction": row["direction"],
            "causal_class": row["causal_class"],
            "confidence": row["confidence"],
            "priority": float(row["priority"]),
            "causal_fraction": float(causal_fraction),
            "intermediate": intermediate,
            "edge_direction": edge_direction,
        })

    query_id = f"query::{query_cyto}"
    register_node(query_id, label_formatter(query_cyto), 0, "rgba(35, 35, 35, 0.90)")
    used_pair_indices = []

    for row_index, row in pairs.iterrows():
        direction_value = row["direction"]
        if direction_value not in _DIRECTION_SIGN:
            raise ValueError(f"Unknown direction: {direction_value!r}")

        secondary = row["secondary"]
        recipients = recipient_weights(row)
        if recipients.empty:
            continue

        if row["causal_class"] == "immediate secondary":
            producers = producer_weights(query_cyto, secondary, direct=True)
            causal_fraction = row["direct_producer_share"]
            if producers.empty or pd.isna(causal_fraction) or causal_fraction <= 0:
                continue
            intermediate = None
            total_flow = float(row["priority"] * causal_fraction)
            final_producer_ids = []

            for producer, producer_share in producers.items():
                producer_id = f"input-producer::{producer}"
                register_node(
                    producer_id,
                    producer,
                    1,
                    "rgba(125, 125, 125, 0.82)",
                    cell_type=producer,
                )
                value = total_flow * producer_share
                add_edge(query_id, producer_id, value, row, "direct stimulation", causal_fraction)
                final_producer_ids.append(producer_id)

        elif row["causal_class"] == "probable tertiary":
            intermediate = row["best_intermediate"]
            if pd.isna(intermediate):
                continue
            first_key = (query_cyto, intermediate)
            second_key = (intermediate, secondary)
            if first_key not in pair_lookup.index or second_key not in pair_lookup.index:
                continue
            first_row = pair_lookup.loc[first_key]
            second_row = pair_lookup.loc[second_key]

            primary_producers = producer_weights(query_cyto, intermediate, direct=True)
            overlap_cells = row.get("best_intermediate_overlap_cells", [])
            final_producers = producer_weights(
                query_cyto,
                secondary,
                direct=False,
                allowed_cells=overlap_cells,
            )
            overlap_share = row.get("best_intermediate_producer_overlap", np.nan)
            causal_fraction = row["indirect_producer_share"] * overlap_share
            if (
                primary_producers.empty
                or final_producers.empty
                or pd.isna(causal_fraction)
                or causal_fraction <= 0
            ):
                continue

            total_flow = float(row["priority"] * causal_fraction)
            intermediate_id = f"cytokine::{intermediate}"
            register_node(
                intermediate_id,
                label_formatter(intermediate),
                2,
                "rgba(120, 70, 145, 0.86)",
            )

            for producer, producer_share in primary_producers.items():
                producer_id = f"input-producer::{producer}"
                register_node(
                    producer_id,
                    producer,
                    1,
                    "rgba(125, 125, 125, 0.82)",
                    cell_type=producer,
                )
                value = total_flow * producer_share
                add_edge(query_id, producer_id, value, row, "direct stimulation", causal_fraction, intermediate)
                add_edge(
                    producer_id, intermediate_id, value, row, "intermediate production",
                    causal_fraction, intermediate, edge_direction=first_row["direction"],
                )

            final_producer_ids = []
            for producer, producer_share in final_producers.items():
                producer_id = f"final-producer::{intermediate}::{producer}"
                register_node(
                    producer_id,
                    producer,
                    3,
                    "rgba(125, 125, 125, 0.82)",
                    cell_type=producer,
                )
                value = total_flow * producer_share
                add_edge(intermediate_id, producer_id, value, row, "intermediate stimulation", causal_fraction, intermediate)
                final_producer_ids.append(producer_id)
        else:
            continue

        # Place direct secondaries in the same cytokine column as intermediates.
        # A shared cytokine (for example IFN-gamma) then has one node that can
        # branch to both direct recipients and an intermediate-responsive cell.
        secondary_id = f"cytokine::{secondary}"
        arrow = "↑" if direction_value == "secondary increased" else "↓"
        secondary_color = (
            "rgba(175, 50, 50, 0.88)"
            if direction_value == "secondary increased"
            else "rgba(50, 85, 175, 0.88)"
        )
        secondary_label = f"{arrow} {label_formatter(secondary)}"
        causal_symbol = _CAUSAL_SYMBOL.get(row["causal_class"], "")
        if show_classification_symbols and causal_symbol:
            # The marker appears inside the final-secondary box, rather than
            # requiring a separate annotation panel in a compact plot.
            secondary_label = f"{causal_symbol} {secondary_label}"
        register_node(
            secondary_id,
            secondary_label,
            2 if row["causal_class"] == "immediate secondary" else 4,
            secondary_color,
        )

        for producer_id in final_producer_ids:
            producer_name = producer_id.rsplit("::", 1)[-1]
            if row["causal_class"] == "immediate secondary":
                weights = producers
            else:
                weights = final_producers
            if producer_name not in weights:
                continue
            add_edge(
                producer_id,
                secondary_id,
                total_flow * weights[producer_name],
                row,
                "secondary production",
                causal_fraction,
                intermediate,
                edge_direction=(
                    second_row["direction"]
                    if row["causal_class"] == "probable tertiary"
                    else row["direction"]
                ),
            )

        for recipient, recipient_share in recipients.items():
            recipient_id = f"recipient::{recipient}"
            register_node(
                recipient_id,
                recipient,
                5,
                "rgba(35, 130, 130, 0.82)",
                cell_type=recipient,
            )
            add_edge(
                secondary_id,
                recipient_id,
                total_flow * recipient_share,
                row,
                "supported downstream response",
                causal_fraction,
                intermediate,
            )
        used_pair_indices.append(row_index)

    if not edge_rows:
        raise ValueError(
            "No complete causal paths remained after producer, receptor, and recipient checks."
        )

    path_table = pd.DataFrame(edge_rows)
    used_pairs = pairs.loc[used_pair_indices].copy()

    node_table = pd.DataFrame(node_rows.values())
    incident_flow = pd.concat([
        path_table.groupby("source")["value"].sum(),
        path_table.groupby("target")["value"].sum(),
    ], axis=1).fillna(0).sum(axis=1)
    node_table["flow"] = node_table["node_id"].map(incident_flow).fillna(0)
    node_table["cell_order"] = node_table["cell_type"].map(cell_order)
    fallback_order = len(cell_order) + node_table.groupby("stage")["flow"].rank(
        ascending=False, method="first"
    )
    node_table["display_order"] = node_table["cell_order"].fillna(fallback_order)
    node_table = node_table.sort_values(
        ["stage", "display_order"], ascending=[True, True]
    ).reset_index(drop=True)

    present_stages = sorted(node_table["stage"].unique())
    # Reserve the right edge for recipient labels, which Plotly draws into the
    # plot area on the left of the final column.  In the common immediate-path
    # layout (input -> producer -> secondary -> recipient), this makes the
    # two middle columns 0.25 and 0.55 instead of evenly spaced 0.34 and 0.66.
    compact_x = {
        2: [0.01, 0.99],
        3: [0.01, 0.42, 0.99],
        4: [0.01, 0.25, 0.55, 0.99],
        5: [0.01, 0.18, 0.42, 0.64, 0.99],
        6: [0.01, 0.16, 0.32, 0.48, 0.66, 0.99],
    }
    x_values = compact_x.get(
        len(present_stages), np.linspace(0.01, 0.99, len(present_stages))
    )
    stage_x = dict(zip(present_stages, x_values))
    node_table["x"] = node_table["stage"].map(stage_x)
    node_table["y"] = 0.5
    for stage, indices in node_table.groupby("stage", sort=False).groups.items():
        indices = list(indices)
        # These positions specify the preferred ALL_CELL_TYPES order.  The
        # ``snap`` layout below may move them apart when their rendered heights
        # would collide (especially in a 4 cm-high panel).
        positions = [0.5] if len(indices) == 1 else np.linspace(0.08, 0.92, len(indices))
        node_table.loc[indices, "y"] = positions

    node_index = pd.Series(node_table.index, index=node_table["node_id"])
    path_table["source_index"] = path_table["source"].map(node_index)
    path_table["target_index"] = path_table["target"].map(node_index)
    customdata = np.column_stack([
        path_table["segment"],
        path_table["secondary"].map(label_formatter),
        path_table["causal_class"],
        path_table["priority"],
        path_table["causal_fraction"],
        path_table["confidence"],
    ])
    link_colors = []
    for _, edge in path_table.iterrows():
        if edge["segment"] in {"direct stimulation", "intermediate stimulation"}:
            link_colors.append("rgba(100, 100, 100, 0.25)")
        elif edge["segment"] == "supported downstream response":
            link_colors.append("rgba(35, 130, 130, 0.35)")
        else:
            link_colors.append(_DIRECTION_COLOR.get(edge["edge_direction"], "rgba(80, 80, 80, 0.35)"))

    figure = go.Figure(go.Sankey(
        # ``fixed`` permits nodes to overlap at the requested y coordinates.
        # ``snap`` retains their columns and preferred order, while moving nodes
        # just enough to prevent collisions.
        arrangement="snap",
        node=dict(
            label=node_table["label"],
            color=node_table["color"],
            x=node_table["x"],
            y=node_table["y"],
            pad=4,
            thickness=10,
            line=dict(color="black", width=0.35),
            hovertemplate="%{label}<extra></extra>",
        ),
        link=dict(
            source=path_table["source_index"],
            target=path_table["target_index"],
            value=path_table["value"],
            color=link_colors,
            customdata=customdata,
            hovertemplate=(
                "%{source.label} → %{target.label}<br>"
                "Step: %{customdata[0]}<br>"
                "Final secondary: %{customdata[1]}<br>"
                "Class: %{customdata[2]}<br>"
                "Confidence: %{customdata[5]}<br>"
                "Pair priority: %{customdata[3]:.2f}<br>"
                "Causally attributed fraction: %{customdata[4]:.2f}<br>"
                "Displayed flow: %{value:.2f}<extra></extra>"
            ),
        ),
    ))

    stage_labels = {
        0: "Input",
        1: "Input-responsive<br>producer",
        2: "First secondary /<br>intermediate",
        3: "Intermediate-responsive<br>producer",
        4: "Downstream<br>secondary",
        5: (
            "Non-direct<br>recipient"
            if recipient_requires_no_query_receptor
            else "Supported<br>recipient"
        ),
    }
    figure.update_layout(
        title=title,
        width=cm_to_px(figsize_cm[0]),
        height=cm_to_px(figsize_cm[1]),
        margin=dict(l=8, r=8, t=35, b=8),
        font=dict(family="Helsana", size=font_size, color="black"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        annotations=[
            dict(
                x=stage_x[stage], y=1.06, xref="paper", yref="paper",
                text=label, showarrow=False, xanchor="center",
                font=dict(size=font_size, color="black"),
            )
            for stage, label in stage_labels.items()
            if stage in set(node_table["stage"])
        ],
    )
    return figure, used_pairs, path_table


def plot_secondary_overview_row_sankey(
    query_cyto,
    pair_table,
    producer_shares_by_pair,
    support_by_pair,
    similarity_by_pair,
    receptor_expression_by_cytokine,
    cell_abundance,
    direction=None,
    minimum_priority=0,
    causal_classes=("immediate secondary", "probable tertiary"),
    confidence_levels=("cross-lineage support", "single-lineage supported"),
    **kwargs,
):
    """Summarize one overview-matrix row as a merged causal network.

    All retained final pairs are plotted together (rather than selecting the
    top ``n_pairs``). Cytokines and input-responsive producer cells are shared
    across branches. A direct branch may therefore run from a first-secondary
    cytokine directly to its non-direct recipients, while the same cytokine can
    also continue through a relay producer to a downstream secondary.

    Returns ``figure, displayed_pairs, path_table, coverage_table``. Entries in
    ``coverage_table`` that are not shown are deliberately not converted into
    causal edges: their overview classification lacks a resolved, supported
    route under the selected filters.
    """
    row = pair_table.loc[
        pair_table["query_cyto"].eq(query_cyto)
        & pair_table["priority"].notna()
    ].copy()
    if direction is not None:
        row = row.loc[row["direction"].eq(direction)]

    class_ok = row["causal_class"].isin(causal_classes)
    confidence_ok = row["confidence"].isin(confidence_levels)
    priority_ok = row["priority"].ge(minimum_priority)
    eligible = class_ok & confidence_ok & priority_ok

    figure, displayed_pairs, path_table = plot_secondary_causal_sankey(
        query_cyto=query_cyto,
        pair_table=pair_table,
        producer_shares_by_pair=producer_shares_by_pair,
        support_by_pair=support_by_pair,
        similarity_by_pair=similarity_by_pair,
        receptor_expression_by_cytokine=receptor_expression_by_cytokine,
        cell_abundance=cell_abundance,
        direction=direction,
        causal_classes=causal_classes,
        confidence_levels=confidence_levels,
        n_pairs=None,
        minimum_priority=minimum_priority,
        title=f"Resolved causal network for {query_cyto}",
        **kwargs,
    )

    displayed_keys = set(zip(displayed_pairs["query_cyto"], displayed_pairs["secondary"]))
    coverage = row.copy()
    coverage["sankey_status"] = "not shown: unresolved or unsupported route"
    coverage.loc[class_ok & confidence_ok & ~priority_ok, "sankey_status"] = (
        "not shown: below priority cutoff"
    )
    coverage.loc[eligible, "sankey_status"] = "not shown: incomplete cell-level route"
    shown_mask = [key in displayed_keys for key in zip(coverage["query_cyto"], coverage["secondary"])]
    coverage.loc[shown_mask, "sankey_status"] = "shown"

    figure.add_annotation(
        x=0.01,
        y=-0.14,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="top",
        showarrow=False,
        text=(
            f"{shown_mask.count(True)} of {len(coverage)} overview pairs shown as "
            "resolved causal paths; remaining entries are retained in coverage_table."
        ),
        font=dict(size=kwargs.get("font_size", 6), color="black"),
    )
    figure.update_layout(margin=dict(l=8, r=8, t=35, b=22))
    return figure, displayed_pairs, path_table, coverage


def plot_secondary_overview_verdict_sankey(
    query_cyto,
    pair_table,
    support_by_pair,
    similarity_by_pair,
    direction=None,
    minimum_recipient_share=0.05,
    minimum_display_flow=0.01,
    cell_type_order=None,
    figsize_cm=(18, 12),
    label_formatter=str,
    font_size=7,
):
    """Summarize every overview-row entry as an explicit causality verdict.

    Unlike a production-path Sankey, its first links mean ``input -> verdict``
    and ``verdict -> observed secondary``.  They are deliberately not claims
    of a biochemical cytokine-to-cytokine interaction.  This lets unresolved,
    unsupported, and ``possibly incidental similarity`` entries remain visible.
    """
    row = pair_table.loc[pair_table["query_cyto"].eq(query_cyto)].copy()
    if direction is not None:
        row = row.loc[row["direction"].eq(direction)]
    if row.empty:
        raise ValueError(f"No overview entries found for {query_cyto!r}.")

    verdict_label = {
        "immediate secondary": "Immediate secondary",
        "probable tertiary": "Probable tertiary",
        "possibly incidental similarity": "Secondary not required",
        "unsupported similarity": "Unsupported similarity",
        "unresolved receptor coverage": "Unresolved receptor coverage",
        "unresolved indirect": "Unresolved indirect route",
        "unresolved": "Unresolved",
        "not evaluable": "Not evaluable",
    }
    verdict_symbol = {
        "immediate secondary": "○",
        "probable tertiary": "△",
        "possibly incidental similarity": "×",
        "unsupported similarity": "!",
        "unresolved receptor coverage": "?",
        "unresolved indirect": "?",
        "unresolved": "?",
        "not evaluable": "·",
    }
    verdict_color = {
        "immediate secondary": "rgba(40, 120, 80, .85)",
        "probable tertiary": "rgba(120, 70, 145, .85)",
        "possibly incidental similarity": "rgba(125, 125, 125, .85)",
        "unsupported similarity": "rgba(170, 170, 170, .85)",
        "unresolved receptor coverage": "rgba(185, 135, 45, .85)",
        "unresolved indirect": "rgba(185, 135, 45, .85)",
        "unresolved": "rgba(185, 135, 45, .85)",
        "not evaluable": "rgba(205, 205, 205, .85)",
    }
    cell_order = {cell: i for i, cell in enumerate(cell_type_order or [])}
    nodes, edges = {}, []

    def node(node_id, label, stage, color, cell_type=None):
        if node_id not in nodes:
            nodes[node_id] = dict(node_id=node_id, label=label, stage=stage, color=color, cell_type=cell_type)

    def edge(source, target, value, segment, row):
        edges.append(dict(source=source, target=target, value=float(value), segment=segment,
                          secondary=row["secondary"], causal_class=row["causal_class"],
                          confidence=row["confidence"], priority=row["priority"], direction=row["direction"]))

    query_id = f"query::{query_cyto}"
    node(query_id, label_formatter(query_cyto), 0, "rgba(35, 35, 35, .9)")
    for _, candidate in row.sort_values("priority", ascending=False, na_position="last").iterrows():
        causal_class = candidate["causal_class"]
        verdict_id = f"verdict::{causal_class}"
        node(verdict_id, verdict_label.get(causal_class, str(causal_class)), 1,
             verdict_color.get(causal_class, "rgba(125, 125, 125, .85)"))
        priority = candidate["priority"]
        flow = max(float(priority) if pd.notna(priority) else 0, minimum_display_flow)
        edge(query_id, verdict_id, flow, "causality verdict", candidate)

        secondary = candidate["secondary"]
        arrow = "↑" if candidate["direction"] == "secondary increased" else "↓"
        symbol = verdict_symbol.get(causal_class, "?")
        secondary_id = f"observed-secondary::{secondary}"
        node(secondary_id, f"{symbol} {arrow} {label_formatter(secondary)}", 2,
             "rgba(175, 50, 50, .88)" if arrow == "↑" else "rgba(50, 85, 175, .88)")
        edge(verdict_id, secondary_id, flow, "observed secondary", candidate)

        support = pd.Series(support_by_pair.get((query_cyto, secondary), pd.Series(dtype=bool)), dtype="boolean").fillna(False).astype(bool)
        similarity = _as_numeric_series(similarity_by_pair.get((query_cyto, secondary)))
        aligned = (_DIRECTION_SIGN.get(candidate["direction"], 0) * similarity).clip(lower=0)
        weights = _trim_and_normalize(aligned.where(support, 0), minimum_recipient_share)
        for recipient, share in weights.items():
            recipient_id = f"compatible-recipient::{recipient}"
            node(recipient_id, recipient, 3, "rgba(95, 145, 145, .82)", cell_type=recipient)
            edge(secondary_id, recipient_id, flow * share, "response-compatible cell", candidate)

    node_table = pd.DataFrame(nodes.values())
    path_table = pd.DataFrame(edges)
    node_table["order"] = node_table["cell_type"].map(cell_order).fillna(
        len(cell_order) + node_table.groupby("stage")["label"].rank(method="dense")
    )
    node_table = node_table.sort_values(["stage", "order", "label"]).reset_index(drop=True)
    stage_x = {0: .01, 1: .29, 2: .58, 3: .99}
    node_table["x"] = node_table["stage"].map(stage_x)
    node_table["y"] = .5
    for stage, indices in node_table.groupby("stage").groups.items():
        indices = list(indices)
        node_table.loc[indices, "y"] = [.5] if len(indices) == 1 else np.linspace(.06, .94, len(indices))
    index = pd.Series(node_table.index, index=node_table.node_id)
    path_table["source_index"] = path_table.source.map(index)
    path_table["target_index"] = path_table.target.map(index)
    colors = ["rgba(100,100,100,.25)" if segment == "causality verdict" else
              _DIRECTION_COLOR.get(direction, "rgba(100,100,100,.3)") if segment == "observed secondary" else
              "rgba(95,145,145,.30)"
              for segment, direction in zip(path_table.segment, path_table.direction)]
    figure = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=node_table.label, color=node_table.color, x=node_table.x, y=node_table.y,
                  pad=4, thickness=10, line=dict(color="black", width=.35)),
        link=dict(source=path_table.source_index, target=path_table.target_index, value=path_table.value,
                  color=colors,
                  hovertemplate="%{source.label} → %{target.label}<br>Step: %{customdata[0]}<br>Verdict: %{customdata[1]}<br>Priority: %{customdata[2]:.2f}<extra></extra>",
                  customdata=np.column_stack([path_table.segment, path_table.causal_class, path_table.priority])),
    ))
    figure.update_layout(
        title=f"Overview verdict summary: {label_formatter(query_cyto)}",
        width=cm_to_px(figsize_cm[0]), height=cm_to_px(figsize_cm[1]),
        margin=dict(l=8, r=8, t=35, b=22), font=dict(family="Helsana", size=font_size, color="black"),
        annotations=[dict(x=.01, y=-.14, xref="paper", yref="paper", xanchor="left", yanchor="top", showarrow=False,
                          text="Grey links encode causality verdicts, not biochemical production. ○ immediate; △ tertiary; × secondary not required; ? unresolved.",
                          font=dict(size=font_size, color="black"))],
    )
    return figure, row, path_table

# Candidate-complete view used by the notebook's plot_secondary_sankey wrapper.
from .secondary_flow_sankey import plot_secondary_flow_sankey
