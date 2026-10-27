"""Compact weighted cell-flow view of secondary cytokine responses."""

from __future__ import annotations

from html import unescape
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Rectangle
import numpy as np
import pandas as pd

from .secondary_flow_sankey import plot_secondary_flow_sankey


_ROUTE_COLORS = {
    "direct": "#313131",
    "relay": "#7A3E9D",
    "unresolved": "#D08A00",
    "unresolved/higher order": "#F2D253",
    "unknown": "#A0A0A0",
}
_SIMPLIFIED_CAUSAL_CLASSES = {
    "shared direct/indirect",
    "probable tertiary",
    "unresolved indirect",
}
_SIMPLIFIED_CAUSAL_LABEL = "unresolved/higher order"
_EMPHASIZED_ROUTE_ALPHA = .86
_PRODUCT_COLORS = ["#2166AC", "#1B9E77", "#B2182B", "#00A6D6", "#D660A8", "#238B45"]
_CELL_ABBREVIATIONS = {
    "CD14 Mono": "CD14",
    "CD16 Mono": "CD16",
    "CD4 T cell": "CD4",
    "CD8 T cell": "CD8",
    "NK CD56hi": "NKhi",
    "NK CD56low": "NKlo",
    "B cell": "B",
}


def _plain_label(value):
    return " ".join(unescape(re.sub(r"<[^>]+>", " ", str(value))).split())


def _cell(node_id):
    return str(node_id).rsplit("::", 1)[-1]


def _route(node_id):
    node_id = str(node_id)
    if node_id.startswith("direct::"):
        return "direct"
    if node_id.startswith("relay::"):
        return "relay"
    if node_id.startswith("indirect::"):
        return "unresolved"
    return "unknown"


def _role(node_id):
    parts = str(node_id).split("::")
    return parts[2] if len(parts) > 3 else "missing"


def _spread_by_score(scores, *, low=.08, high=.92, tie_order=None):
    tie_order = tie_order or {}
    ordered = sorted(scores, key=lambda name: (-scores[name], tie_order.get(name, len(tie_order)), name))
    positions = np.linspace(high, low, len(ordered)) if len(ordered) > 1 else np.array([.5])
    return dict(zip(ordered, positions))


def _curve(ax, x0, y0, x1, y1, *, color, width, alpha, zorder=1):
    bend = (x1 - x0) * .48
    path = Path(
        [(x0, y0), (x0 + bend, y0), (x1 - bend, y1), (x1, y1)],
        [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4],
    )
    ax.add_patch(PathPatch(path, facecolor="none", edgecolor=color, linewidth=width,
                           alpha=alpha, capstyle="round", zorder=zorder))


def _port_offsets(frame, group_column, order_values, spacing):
    """Assign centred, non-coincident ports ordered from top to bottom."""
    offsets = {}
    for _, group in frame.groupby(group_column, sort=False):
        ordered = group.assign(_order=group.index.map(order_values)).sort_values(
            ["_order", "value"], ascending=[False, False], kind="stable"
        )
        count = len(ordered)
        values = ((count - 1) / 2 - np.arange(count)) * spacing
        offsets.update(dict(zip(ordered.index, values)))
    return offsets


def _all_similarity_values(
    query_cyto,
    pairs,
    similarity_by_pair,
    support_by_pair,
    producer_details_by_pair,
    pair_details_by_pair,
    *,
    receptor_min,
    recipient_role="all",
):
    """Return unthresholded Pearson r and direction-aligned edge similarities."""
    signs = {"secondary increased": 1, "secondary reduced": -1}
    role_by_class = {"direct": "facultative", "indirect": "obligate", "missing": "unknown"}
    records = []
    for _, pair in pairs.iterrows():
        sign = signs.get(pair.direction)
        if sign is None:
            raise ValueError(f"Unknown direction {pair.direction!r}")
        key = (query_cyto, pair.secondary)
        similarities = pd.to_numeric(
            pd.Series(similarity_by_pair.get(key, {}), dtype=float), errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)
        support = pd.Series(support_by_pair.get(key, {}), dtype="boolean").reindex(
            similarities.index
        )
        pair_details = pair_details_by_pair.get(key, {})
        if "secondary_receptor" not in pair_details:
            raise ValueError(
                f"Missing secondary_receptor values in pair_details_by_pair[{key!r}]."
            )
        secondary_receptor = pd.to_numeric(
            pd.Series(pair_details["secondary_receptor"]), errors="coerce"
        ).reindex(similarities.index)
        receptor_expressed = secondary_receptor.ge(receptor_min).fillna(False)
        receptor_classes = pd.Series(
            producer_details_by_pair.get(key, {}).get("producer_receptor_classes", {}),
            dtype=object,
        ).reindex(similarities.index).fillna("missing")
        roles = receptor_classes.map(role_by_class).fillna("unknown")
        for cell, similarity in similarities.dropna().items():
            role = roles.loc[cell]
            if recipient_role != "all" and role != recipient_role:
                continue
            records.append({
                "query_cyto": query_cyto,
                "secondary": str(pair.secondary),
                "product": str(pair.secondary),
                "cell": str(cell),
                "similarity": float(similarity),
                "raw_similarity": float(sign * similarity),
                "supported": bool(support.loc[cell]) if pd.notna(support.loc[cell]) else False,
                "receptor_expressed": bool(receptor_expressed.loc[cell]),
                "role": role,
            })
    return pd.DataFrame.from_records(records, columns=[
        "query_cyto", "secondary", "product", "cell", "similarity",
        "raw_similarity", "supported", "receptor_expressed", "role",
    ])


def _network_from_paths(
    query_cyto,
    pairs,
    paths,
    *,
    cell_type_order=None,
    label_formatter=str,
    figsize_cm=(6, 2.5),
    font_size=6,
    minimum_producer_share=.10,
    minimum_recipient_similarity=.30,
    path_width=.85,
    path_alpha=.62,
    port_spacing=.018,
    left_bar_height=.035,
    right_bar_height=.055,
    show_bar_spines=True,
    title=None,
    show_legend=True,
    all_similarities=None,
    similarity_panel_width_cm=.60,
    simplified=False,
):
    """Render already-constructed flow evidence as a compact broadcast network."""
    if any(not np.isfinite(value) or value < 0 or value > 1 for value in
           (minimum_producer_share, minimum_recipient_similarity)):
        raise ValueError("Cutoffs must be finite fractions between 0 and 1.")
    if len(figsize_cm) != 2 or any(not np.isfinite(value) or value <= 0 for value in figsize_cm):
        raise ValueError("figsize_cm must contain two finite positive values.")
    if not np.isfinite(font_size) or font_size < 5.5:
        raise ValueError("font_size must be at least 5.5 pt.")
    font_size = float(font_size)
    small_font_size = max(5.5, font_size - .5)
    if not np.isfinite(path_width) or path_width <= 0:
        raise ValueError("path_width must be positive.")
    if not np.isfinite(path_alpha) or not 0 <= path_alpha <= 1:
        raise ValueError("path_alpha must be between 0 and 1.")
    if not np.isfinite(port_spacing) or port_spacing <= 0:
        raise ValueError("port_spacing must be positive.")
    if not np.isfinite(left_bar_height) or left_bar_height <= 0:
        raise ValueError("left_bar_height must be positive.")
    if not np.isfinite(right_bar_height) or right_bar_height <= 0:
        raise ValueError("right_bar_height must be positive.")
    if not np.isfinite(similarity_panel_width_cm) or similarity_panel_width_cm <= 0:
        raise ValueError("similarity_panel_width_cm must be positive.")

    paths = paths.copy()
    paths["value"] = pd.to_numeric(paths.value, errors="coerce")
    paths["raw_similarity"] = pd.to_numeric(paths.raw_similarity, errors="coerce")
    products = pairs.secondary.astype(str).tolist()
    direction_signs = {"secondary increased": 1., "secondary reduced": -1.}
    product_sign = {}
    for _, pair in pairs.iterrows():
        if pair.direction not in direction_signs:
            raise ValueError(f"Unknown direction {pair.direction!r}")
        product_sign[str(pair.secondary)] = direction_signs[pair.direction]
    product_y = dict(zip(products, np.linspace(.80, .20, len(products)) if len(products) > 1 else [.5]))
    product_color = {product: _PRODUCT_COLORS[index % len(_PRODUCT_COLORS)]
                     for index, product in enumerate(products)}

    production = paths.loc[
        paths.target.isin([f"product::{product}" for product in products])
    ].copy()
    production["product"] = production.secondary.astype(str)
    production["cell"] = production.source.map(_cell)
    production["route"] = production.source.map(_route)
    production["display_route"] = production["route"]
    if simplified:
        higher_order_products = set(
            pairs.loc[
                pairs["causal_class"].eq(_SIMPLIFIED_CAUSAL_LABEL), "secondary"
            ].astype(str)
        )
        production.loc[
            production["route"].ne("direct")
            | production["product"].isin(higher_order_products),
            "display_route",
        ] = _SIMPLIFIED_CAUSAL_LABEL
        # Route components that now have the same meaning must become one path,
        # not multiple coincident connections. Aggregate before thresholding so
        # the cutoff applies to the displayed, combined contribution.
        collapsed_rows = []
        for _, group in production.groupby(
            ["product", "cell", "display_route"], sort=False, dropna=False
        ):
            row = group.iloc[0].copy()
            row["value"] = group["value"].sum(min_count=1)
            row["route_components"] = tuple(dict.fromkeys(group["route"].astype(str)))
            row["intermediate_components"] = tuple(
                dict.fromkeys(group["intermediate"].dropna().astype(str))
            )
            if len(group) > 1:
                row["source"] = f"combined::{row['cell']}"
                row["route"] = row["display_route"]
                row["intermediate"] = pd.NA
            collapsed_rows.append(row)
        if collapsed_rows:
            production = pd.DataFrame(collapsed_rows)
        else:
            production["route_components"] = pd.Series(index=production.index, dtype=object)
            production["intermediate_components"] = pd.Series(index=production.index, dtype=object)

    production = production.loc[
        production["value"].ge(minimum_producer_share)
    ].copy()
    production["production_direction_sign"] = production["product"].map(product_sign)
    production["signed_production_fraction"] = (
        production["value"] * production["production_direction_sign"]
    )

    response = paths.loc[
        paths.segment.eq("supported response similarity")
        & paths.raw_similarity.ge(minimum_recipient_similarity)
        & paths.secondary.astype(str).isin(products)
    ].copy()
    response["product"] = response.secondary.astype(str)
    response["cell"] = response.target.map(_cell)
    response["role"] = response.target.map(_role)

    if all_similarities is None:
        # Compatibility for direct callers. The public entry point supplies the
        # complete, unthresholded values from similarity_by_pair.
        all_similarities = response.copy()
        all_similarities["similarity"] = (
            all_similarities["raw_similarity"]
            * all_similarities["product"].map(product_sign)
        )
        all_similarities = all_similarities[
            ["query_cyto", "secondary", "product", "cell", "similarity",
             "raw_similarity", "role"]
        ].copy()
        all_similarities["supported"] = True
    else:
        all_similarities = all_similarities.copy()
    if "receptor_expressed" in all_similarities:
        all_similarities = all_similarities.loc[
            all_similarities["receptor_expressed"].fillna(False).astype(bool)
        ].copy()
    all_similarities["raw_similarity"] = pd.to_numeric(
        all_similarities.raw_similarity, errors="coerce"
    )
    all_similarities["similarity"] = pd.to_numeric(
        all_similarities.similarity, errors="coerce"
    )

    producer_scores = {
        cell: float(np.average(group["product"].map(product_y), weights=group.value))
        for cell, group in production.groupby("cell")
    }
    tie_order = {cell: index for index, cell in enumerate(cell_type_order or [])}
    producer_y = _spread_by_score(producer_scores, tie_order=tie_order)
    # A recipient node exists only if at least one displayed cytokine connection
    # reaches it. Group nodes by whether the query-cytokine receptor is expressed,
    # retaining the requested cell order within each group.
    observed_cells = response.cell.astype(str).drop_duplicates().tolist()
    base_cell_order = [cell for cell in (cell_type_order or []) if cell in observed_cells]
    base_cell_order += [cell for cell in observed_cells if cell not in base_cell_order]
    receptor_status_by_cell = {}
    for cell, group in response.groupby("cell", sort=False):
        roles = set(group["role"])
        role = next(iter(roles)) if len(roles) == 1 else "missing"
        receptor_status_by_cell[str(cell)] = {
            "direct": "yes", "indirect": "no",
        }.get(role, "unknown")
    status_order = ("yes", "no", "unknown")
    recipient_groups = {
        status: [
            cell for cell in base_cell_order
            if receptor_status_by_cell.get(cell, "unknown") == status
        ]
        for status in status_order
    }
    ordered_cells = [cell for status in status_order for cell in recipient_groups[status]]
    # A fractional extra slot makes the receptor-status grouping visible without
    # wasting the limited vertical space of this compact figure.
    position_slots = {}
    cursor = 0.
    nonempty_groups = [status for status in status_order if recipient_groups[status]]
    for group_index, status in enumerate(nonempty_groups):
        for cell in recipient_groups[status]:
            position_slots[cell] = cursor
            cursor += 1.
        if group_index < len(nonempty_groups) - 1:
            cursor += .55
    maximum_slot = max(position_slots.values(), default=0.)
    recipient_y = {
        cell: .5 if maximum_slot == 0 else .92 - .84 * slot / maximum_slot
        for cell, slot in position_slots.items()
    }

    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size,
        "xtick.labelsize": small_font_size,
        "ytick.labelsize": small_font_size,
        "legend.fontsize": small_font_size,
        "legend.title_fontsize": small_font_size,
        "svg.fonttype": "none",
    })
    # ``figsize_cm`` is the size of the complete figure.  The virtual width is
    # used only to divide that fixed canvas between the network and the compact
    # similarity panels.
    virtual_width_cm = figsize_cm[0] + similarity_panel_width_cm * len(products)
    network_fraction = figsize_cm[0] / virtual_width_cm
    fig = plt.figure(figsize=(figsize_cm[0] / 2.54, figsize_cm[1] / 2.54))
    top = .88 if title is None else .82
    bottom = .24 if show_legend else .10
    # Keep the requested network canvas width and append one compact similarity
    # axis per selected cytokine.
    ax = fig.add_axes([.10 * network_fraction, bottom, .88 * network_fraction, top - bottom])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    x_producer, x_product, x_recipient = .25, .54, .90

    production_source_order = {index: product_y[row["product"]] for index, row in production.iterrows()}
    production_target_order = {index: producer_y[row["cell"]] for index, row in production.iterrows()}
    producer_ports = _port_offsets(production, "cell", production_source_order, port_spacing)
    product_in_ports = _port_offsets(production, "product", production_target_order, port_spacing)
    for index, row in production.sort_values(["product", "value"], ascending=[True, False]).iterrows():
        route_alpha = (
            max(path_alpha, _EMPHASIZED_ROUTE_ALPHA)
            if row["display_route"] in {"direct", "unresolved", _SIMPLIFIED_CAUSAL_LABEL}
            else path_alpha
        )
        _curve(ax, x_producer + .012, producer_y[row["cell"]] + producer_ports[index],
               x_product - .006, product_y[row["product"]] + product_in_ports[index],
               color=_ROUTE_COLORS[row["display_route"]], width=path_width, alpha=route_alpha)
        if row["display_route"] == "relay" and pd.notna(row["intermediate"]):
            ax.text(.39, (producer_y[row["cell"]] + product_y[row["product"]]) / 2,
                    "via " + _plain_label(label_formatter(str(row["intermediate"]))), fontsize=small_font_size,
                    color=_ROUTE_COLORS["relay"], ha="center", va="center")

    # Port order follows the opposite endpoint's vertical position. Within a
    # hub this prevents avoidable crossings and ensures no paths are coincident.
    response_for_ports = response.assign(value=response.raw_similarity)
    response_source_order = {index: recipient_y[row["cell"]] for index, row in response.iterrows()}
    response_target_order = {index: product_y[row["product"]] for index, row in response.iterrows()}
    product_out_ports = _port_offsets(response_for_ports, "product", response_source_order, port_spacing)
    recipient_ports = _port_offsets(response_for_ports, "cell", response_target_order, port_spacing)
    for index, row in response.sort_values(["product", "raw_similarity"], ascending=[True, False]).iterrows():
        _curve(ax, x_product + .006, product_y[row["product"]] + product_out_ports[index],
               x_recipient - .018, recipient_y[row["cell"]] + recipient_ports[index],
               color=product_color[row["product"]], width=path_width, alpha=path_alpha)

    # Compact quantitative summary for producers. Cytokines are shown as
    # adjacent bars rather than stacked segments and normalized to the largest
    # displayed cell--cytokine contribution.
    producer_bars = production.groupby(
        ["cell", "product"], as_index=False, sort=False
    )["value"].sum()
    producer_bars["direction_sign"] = producer_bars["product"].map(product_sign)
    producer_bars["signed_value"] = (
        producer_bars["value"] * producer_bars["direction_sign"]
    )
    producer_max = (
        float(producer_bars["signed_value"].abs().max())
        if not producer_bars.empty else 1.
    )
    if not np.isfinite(producer_max) or producer_max <= 0:
        producer_max = 1.
    has_downregulation = bool(producer_bars["signed_value"].lt(0).any())
    bar_spine_linewidth = .75
    producer_bar_right = .145
    producer_bar_left = .005 if has_downregulation else .035
    # Positive-only plots begin at zero and use the full available width. When
    # decreased production is present, reserve equal-width negative and
    # positive halves around a centred zero line.
    producer_bar_zero = .075 if has_downregulation else producer_bar_left
    producer_bar_width = (
        min(
            producer_bar_zero - producer_bar_left,
            producer_bar_right - producer_bar_zero,
        )
        if has_downregulation
        else producer_bar_right - producer_bar_zero
    )

    # Build a local lane set for each cell from only the products it actually
    # produces. Heights shrink only when needed to remain inside the cell's
    # region (bounded by the midpoints to neighbouring producer nodes).
    producer_bar_specs = []
    effective_left_bar_heights = {}
    cells_by_y = sorted(producer_y, key=producer_y.get)
    for cell_index, cell in enumerate(cells_by_y):
        y = producer_y[cell]
        lower_bound = (
            .005 if cell_index == 0
            else (producer_y[cells_by_y[cell_index - 1]] + y) / 2
        )
        upper_bound = (
            .995 if cell_index == len(cells_by_y) - 1
            else (y + producer_y[cells_by_y[cell_index + 1]]) / 2
        )
        cell_rows = producer_bars.loc[producer_bars.cell.eq(cell)].copy()
        cell_rows["_product_order"] = cell_rows["product"].map(
            {product: index for index, product in enumerate(products)}
        )
        cell_rows = cell_rows.sort_values("_product_order")
        count = len(cell_rows)
        if count == 0:
            continue
        half_region = .90 * min(y - lower_bound, upper_bound - y)
        half_height_units = count / 2
        effective_height = min(left_bar_height, half_region / half_height_units)
        effective_left_bar_heights[cell] = effective_height
        for local_index, (_, row) in enumerate(cell_rows.iterrows()):
            y_offset = ((count - 1) / 2 - local_index) * effective_height
            signed_value = float(row["signed_value"])
            width = producer_bar_width * np.clip(
                abs(signed_value), 0., producer_max
            ) / producer_max
            producer_bar_specs.append({
                "cell": cell,
                "product": row["product"],
                "x": producer_bar_zero if signed_value >= 0 else producer_bar_zero - width,
                "y": y + y_offset - effective_height / 2,
                "width": width,
                "height": effective_height,
                "signed_value": signed_value,
            })

    if show_bar_spines:
        # Size the spine to the bars actually present.
        if producer_bar_specs:
            bar_bottom = max(.005, min(spec["y"] for spec in producer_bar_specs) - .008)
            bar_top = min(.995, max(
                spec["y"] + spec["height"] for spec in producer_bar_specs
            ) + .008)
        else:
            bar_bottom, bar_top = .045, .955
        ax.plot([producer_bar_zero, producer_bar_zero], [bar_bottom, bar_top],
                color="#404040", linewidth=bar_spine_linewidth,
                solid_capstyle="butt", zorder=4)
        ax.plot([producer_bar_left, producer_bar_right], [bar_bottom, bar_bottom],
                color="#404040", linewidth=bar_spine_linewidth,
                solid_capstyle="butt", zorder=4)
        if has_downregulation:
            ax.plot([producer_bar_left, producer_bar_left],
                    [bar_bottom, bar_bottom + .012], color="#404040",
                    linewidth=bar_spine_linewidth, solid_capstyle="butt", zorder=4)
            ax.text(producer_bar_left, bar_bottom - .015, "−", ha="center", va="top",
                    fontsize=small_font_size, color="#404040")
        ax.text(producer_bar_right, bar_bottom - .015, "+", ha="center", va="top",
                fontsize=small_font_size, color="#404040")

    for spec in producer_bar_specs:
        ax.add_patch(Rectangle((spec["x"], spec["y"]),
                               spec["width"], spec["height"],
                               facecolor=product_color[spec["product"]], edgecolor="none",
                               zorder=3))

    for cell, y in producer_y.items():
        ax.scatter([x_producer], [y], s=13, marker="o", facecolor="#F8F8F8",
                   edgecolor="#444444", linewidth=.55, zorder=5)
        ax.text(x_producer - .018, y, _CELL_ABBREVIATIONS.get(cell, cell), ha="right", va="center",
                fontsize=font_size, color="#252525")

    for product in products:
        y = product_y[product]
        has_response = bool(response["product"].eq(product).any())
        incoming_span = max([abs(product_in_ports[index]) for index in production.index
                             if production.loc[index, "product"] == product] or [0])
        outgoing_span = max([abs(product_out_ports[index]) for index in response.index
                             if response.loc[index, "product"] == product] or [0])
        hub_half_height = max(incoming_span, outgoing_span, .022)
        ax.plot([x_product, x_product], [y - hub_half_height, y + hub_half_height],
                color=product_color[product], linewidth=4.2, solid_capstyle="round", zorder=6)
        product_label = _plain_label(label_formatter(product))
        ax.text(x_product, y + hub_half_height + .025, product_label, ha="center", va="bottom",
                fontsize=font_size, color="#202020")
        if not has_response:
            ax.text(x_product + .035, y, "no support", ha="left", va="center",
                    fontsize=small_font_size, color="#777777")

    for cell, y in recipient_y.items():
        receptor_status = receptor_status_by_cell.get(cell, "unknown")
        if receptor_status == "no":
            marker, face, edge = "s", "#303030", "#303030"
        elif receptor_status == "yes":
            marker, face, edge = "o", "none", "#303030"
        else:
            marker, face, edge = "D", "none", "#777777"
        ax.scatter([x_recipient], [y], s=14, marker=marker, facecolor=face,
                   edgecolor=edge, linewidth=.75, zorder=5)
        ax.text(x_recipient + .018, y, _CELL_ABBREVIATIONS.get(cell, cell), ha="left", va="center",
                fontsize=font_size, color="#252525")

    query_label = _plain_label(label_formatter(str(query_cyto)))
    ax.text(-.095, .5, f"{query_label} response producer", transform=ax.transAxes,
            rotation=90, ha="center", va="center", fontsize=font_size, clip_on=False)
    ax.text(x_producer, 1.035, "producer", ha="center", va="bottom",
            fontsize=font_size)
    ax.text(x_product, 1.035, "cytokine", ha="center", va="bottom", fontsize=font_size)
    ax.text(x_recipient, 1.035, "recipient", ha="center", va="bottom", fontsize=font_size)

    # Independent panels show ordinary Pearson r rather than direction-aligned
    # values. Every axis has the same numerical span and physical width, so bar
    # length has one meaning throughout. One-sided panels shift that common span
    # to start or end at zero rather than reserving space on the unused side.
    panel_left = (figsize_cm[0] + .02) / virtual_width_cm
    panel_right = .99
    panel_gap = .04 / virtual_width_cm
    panel_width = (panel_right - panel_left - panel_gap * (len(products) - 1)) / len(products)
    similarity_values_by_product = {}
    required_similarity_ranges = {}
    for product in products:
        values = all_similarities.loc[
            all_similarities["product"].eq(product)
        ].set_index("cell")["similarity"].reindex(ordered_cells)
        similarity_values_by_product[product] = values
        finite_values = values.dropna().to_numpy(dtype=float)
        if finite_values.size:
            required_min = min(0., np.floor(finite_values.min() * 10) / 10)
            required_max = max(0., np.ceil(finite_values.max() * 10) / 10)
        else:
            required_min, required_max = 0., .1
        if np.isclose(required_min, required_max):
            required_max = required_min + .1
        required_similarity_ranges[product] = (required_min, required_max)

    similarity_span = max(
        maximum - minimum for minimum, maximum in required_similarity_ranges.values()
    )
    similarity_ranges = {}
    for product_index, product in enumerate(products):
        similarity_ax = fig.add_axes([
            panel_left + product_index * (panel_width + panel_gap),
            bottom,
            panel_width,
            top - bottom,
        ])
        values = similarity_values_by_product[product]
        required_min, required_max = required_similarity_ranges[product]
        if required_max <= 0:
            similarity_min, similarity_max = -similarity_span, 0.
        elif required_min >= 0:
            similarity_min, similarity_max = 0., similarity_span
        else:
            unused_span = similarity_span - (required_max - required_min)
            lower_padding = np.floor(unused_span * 5) / 10
            similarity_min = required_min - lower_padding
            similarity_max = similarity_min + similarity_span
        similarity_ticks = sorted(set([similarity_min, 0., similarity_max]))
        similarity_ranges[product] = (similarity_min, similarity_max)
        y_positions = np.array([recipient_y[cell] for cell in ordered_cells])
        valid = values.notna().to_numpy()
        similarity_ax.barh(
            y_positions[valid], values.to_numpy()[valid], height=right_bar_height,
            color=product_color[product], edgecolor="none",
        )
        similarity_ax.set_xlim(similarity_min, similarity_max)
        similarity_ax.set_ylim(0, 1)
        similarity_ax.set_xticks(similarity_ticks)
        similarity_ax.tick_params(
            axis="x", labelbottom=product_index == 0, length=1.5,
            width=bar_spine_linewidth, labelsize=small_font_size, pad=1,
        )
        if product_index == 0:
            similarity_ax.set_xlabel(f"Pearson r to {query_label}", fontsize=font_size, labelpad=1)
        similarity_ax.set_yticks([])
        product_label = _plain_label(label_formatter(product))
        if len(product_label) > 8 and "-" in product_label:
            product_label = product_label.replace("-", "-\n", 1)
        similarity_ax.set_title(
            product_label, fontsize=font_size,
            color=product_color[product], pad=2,
        )
        similarity_ax.spines["top"].set_visible(False)
        similarity_ax.spines["right"].set_visible(False)
        # Use a real Matplotlib spine at r=0. An axvline at an exact x-limit can
        # be clipped or hidden behind the bars, especially in exported SVGs.
        similarity_ax.spines["left"].set_position(("data", 0.))
        similarity_ax.spines["left"].set_visible(show_bar_spines)
        similarity_ax.spines["left"].set_color("#404040")
        similarity_ax.spines["left"].set_linewidth(bar_spine_linewidth)
        similarity_ax.spines["left"].set_zorder(5)
        similarity_ax.spines["bottom"].set_visible(show_bar_spines)
        similarity_ax.spines["bottom"].set_color("#404040")
        similarity_ax.spines["bottom"].set_linewidth(bar_spine_linewidth)
        similarity_ax.spines["bottom"].set_zorder(5)
    if title:
        fig.suptitle(title, fontsize=font_size, y=.99)
    if show_legend:
        route_legend = (
            [
                mpl.lines.Line2D([], [], color=_ROUTE_COLORS["direct"], lw=2, label="direct"),
                mpl.lines.Line2D([], [], color=_ROUTE_COLORS[_SIMPLIFIED_CAUSAL_LABEL], lw=2,
                                 label="unresolved/higher order"),
            ]
            if simplified else
            [
                mpl.lines.Line2D([], [], color=_ROUTE_COLORS["direct"], lw=2, label="direct"),
                mpl.lines.Line2D([], [], color=_ROUTE_COLORS["relay"], lw=2, label="relay"),
                mpl.lines.Line2D([], [], color=_ROUTE_COLORS["unresolved"], lw=2, label="unresolved"),
            ]
        )
        receptor_legend = [
            mpl.lines.Line2D([], [], marker="o", markerfacecolor="none", markeredgecolor="#303030",
                             linestyle="none", markersize=3.2, label="yes"),
            mpl.lines.Line2D([], [], marker="s", markerfacecolor="#303030", markeredgecolor="#303030",
                             linestyle="none", markersize=3., label="no"),
        ]
        fig.legend(handles=route_legend, loc="lower center",
                   bbox_to_anchor=(network_fraction * .28, .005),
                   frameon=False, ncol=len(route_legend), fontsize=small_font_size,
                   handlelength=1.15, handletextpad=.35, columnspacing=.65,
                   borderaxespad=0)
        fig.legend(handles=receptor_legend, title=f"{query_label} receptor",
                   loc="lower center", bbox_to_anchor=(network_fraction * .72, .002),
                   frameon=False, ncol=2, fontsize=small_font_size,
                   title_fontsize=small_font_size, handlelength=1.,
                   handletextpad=.3, columnspacing=.65, borderaxespad=0)

    displayed = pd.concat([
        production.assign(
            view_side="production",
            displayed_weight=production.signed_production_fraction,
        ),
        response.assign(view_side="response", displayed_weight=response.raw_similarity),
    ], ignore_index=True, sort=False)
    displayed.attrs["expanded_paths"] = paths
    displayed.attrs["cutoffs"] = {
        "producer": minimum_producer_share,
        "recipient_similarity": minimum_recipient_similarity,
    }
    displayed.attrs["bar_maxima"] = {
        "producer_share": producer_max,
        "recipient_similarity_ranges": similarity_ranges,
        "recipient_pearson_ranges": similarity_ranges,
        "recipient_similarity_span": similarity_span,
    }
    displayed.attrs["bar_heights"] = {
        "left_requested": left_bar_height,
        "left_effective_by_cell": effective_left_bar_heights,
        "right": right_bar_height,
    }
    displayed.attrs["producer_bar_layout"] = "signed locally centred parallel vertical lanes"
    displayed.attrs["producer_bar_order_top_to_bottom"] = list(products)
    displayed.attrs["producer_bar_geometry"] = pd.DataFrame(producer_bar_specs)
    displayed.attrs["producer_bar_range"] = (
        (-producer_max, producer_max)
        if has_downregulation else (0., producer_max)
    )
    displayed.attrs["producer_bar_zero_x"] = producer_bar_zero
    displayed.attrs["all_similarities"] = all_similarities
    displayed.attrs["recipient_bar_layout"] = "one panel per secondary cytokine"
    displayed.attrs["recipient_bar_metric"] = "Pearson r"
    displayed.attrs["recipient_order_top_to_bottom"] = ordered_cells
    displayed.attrs["recipient_receptor_status"] = receptor_status_by_cell
    displayed.attrs["recipient_group_gap_slots"] = .55
    displayed.attrs["emphasized_route_alpha"] = _EMPHASIZED_ROUTE_ALPHA
    displayed.attrs["simplified"] = simplified
    return fig, pairs, displayed


def plot_secondary_cell_flow_network(
    query_cyto,
    *,
    pair_table,
    pair_details_by_pair,
    producer_shares_by_pair,
    producer_details_by_pair,
    support_by_pair,
    similarity_by_pair,
    config,
    downstream_cytokines=None,
    secondary=None,
    n_pairs=None,
    mode="all",
    direction=None,
    recipient_role="all",
    similarity_reference=.8,
    cell_type_order=None,
    label_formatter=str,
    figsize_cm=(6, 2.5),
    font_size=6,
    minimum_producer_share=.10,
    minimum_recipient_similarity=.30,
    path_width=.85,
    path_alpha=.62,
    port_spacing=.018,
    left_bar_height=.035,
    right_bar_height=.055,
    show_bar_spines=True,
    title=None,
    show_legend=True,
    similarity_panel_width_cm=.60,
    simplified=False,
):
    """Plot producer-cell → cytokine → recipient-cell evidence as a network.

    Paths have constant width and encode topology/category only. Parallel signed
    bars at producer cells show per-candidate production shares. Separate bar graphs on
    the right show every finite Pearson correlation for each chosen
    secondary cytokine, including unsupported and below-edge-cutoff values.
    Producer fractions extend right for increased production and left for
    decreased production.
    Similarities are shown only where the corresponding secondary receptor in
    ``pair_details_by_pair`` reaches ``config.receptor_min``. Recipient nodes
    without any displayed cytokine connection are omitted.
    ``figsize_cm`` sizes the complete figure. ``similarity_panel_width_cm`` sets
    the relative space assigned to each similarity panel within that fixed canvas.
    ``left_bar_height`` and ``right_bar_height`` are in their respective axes'
    y-coordinate units. ``font_size`` sets the primary typography and must be at
    least 5.5 pt; compact annotations use at most 0.5 pt less, with the same
    5.5 pt floor.

    Returns ``(figure, displayed_pairs, displayed_paths)``.

    With ``simplified=True``, shared secondary/tertiary, plausible tertiary,
    and unresolved tertiary pairs are displayed as one ``unresolved/higher
    order`` route category.
    """
    if not np.isfinite(font_size) or font_size < 5.5:
        raise ValueError("font_size must be at least 5.5 pt.")
    font_size = float(font_size)

    audit_figure, pairs, paths = plot_secondary_flow_sankey(
        query_cyto,
        pair_table=pair_table,
        producer_shares_by_pair=producer_shares_by_pair,
        producer_details_by_pair=producer_details_by_pair,
        support_by_pair=support_by_pair,
        similarity_by_pair=similarity_by_pair,
        config=config,
        mode=mode,
        direction=direction,
        secondary=secondary,
        downstream_cytokines=downstream_cytokines,
        n_pairs=n_pairs,
        recipient_role=recipient_role,
        similarity_reference=similarity_reference,
        cell_type_order=cell_type_order,
        label_formatter=label_formatter,
        layout="expanded",
        compact=False,
        figsize_cm=figsize_cm,
        font_size=font_size,
        title=title,
    )
    del audit_figure
    if simplified:
        pairs = pairs.copy()
        pairs["causal_class_original"] = pairs["causal_class"]
        pairs["causal_class"] = pairs["causal_class"].replace(
            {label: _SIMPLIFIED_CAUSAL_LABEL for label in _SIMPLIFIED_CAUSAL_CLASSES}
        )
        paths = paths.copy()
        if "causal_class" in paths:
            paths["causal_class_original"] = paths["causal_class"]
            paths["causal_class"] = paths["causal_class"].replace(
                {label: _SIMPLIFIED_CAUSAL_LABEL for label in _SIMPLIFIED_CAUSAL_CLASSES}
            )
    all_similarities = _all_similarity_values(
        query_cyto,
        pairs,
        similarity_by_pair,
        support_by_pair,
        producer_details_by_pair,
        pair_details_by_pair,
        receptor_min=config.receptor_min,
        recipient_role=recipient_role,
    )
    return _network_from_paths(
        query_cyto,
        pairs,
        paths,
        cell_type_order=cell_type_order,
        label_formatter=label_formatter,
        figsize_cm=figsize_cm,
        font_size=font_size,
        minimum_producer_share=minimum_producer_share,
        minimum_recipient_similarity=minimum_recipient_similarity,
        path_width=path_width,
        path_alpha=path_alpha,
        port_spacing=port_spacing,
        left_bar_height=left_bar_height,
        right_bar_height=right_bar_height,
        show_bar_spines=show_bar_spines,
        title=title,
        show_legend=show_legend,
        all_similarities=all_similarities,
        similarity_panel_width_cm=similarity_panel_width_cm,
        simplified=simplified,
    )
