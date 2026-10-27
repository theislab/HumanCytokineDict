import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib as mpl
from scipy.stats import mannwhitneyu

def cm_to_inches(cm_tuple):
    return tuple(x / 2.54 for x in cm_tuple)

def setup_plotting_environment():
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.rcdefaults()
    mpl.font_manager.fontManager.ttflist = [
        font_entry
        for font_entry in mpl.font_manager.fontManager.ttflist
        if "helvetica-compressed" not in font_entry.fname.lower()
    ]
    plt.rcParams.update({
        "font.size": 6,
        "font.family": "Helvetica",
        "font.style": "normal",
        "font.weight": "normal",
        "font.stretch": "normal",
        "axes.titleweight": "normal",
        "axes.labelweight": "normal",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

def plot_overview_labels_along_y_axis(fig, ax, labels, vals_per_label, x_offset, no_y_vals,
                                      fontsize=7, fontweight="bold", color="black", linewidth=1, stagger=0,
                                      rotation=90):
    """
    Plot labels along the y-axis with vertical lines to group related items.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure object containing the plot
    ax : matplotlib.axes.Axes
        The axes object to plot on
    labels : list or array-like
        List of label strings to display along the y-axis
    vals_per_label: dict or pandas.Series
        Dictionary/Series mapping each label to the number of values it represents.
        Keys should match entries in 'labels'
    x_offset : float
        Horizontal position offset for placing the labels (negative values place left of y-axis)
    no_y_vals : int
        Total number of y-axis values/positions to distribute labels across
    fontsize : int, optional (default=7)
        Font size for the label text
    fontweight : str, optional (default="bold")
        Font weight for the label text ("bold", "normal", etc.)
    color : str, dict, optional (default="black")
        Color for both text and lines
    linewidth : float, optional (default=1)
        Width of the vertical grouping lines
    stagger : float, optional (default=0)
        Horizontal staggering offset for alternating labels (to avoid overlap)
    rotation : float, optional (default=90)
        Rotation angle for the label text in degrees
    """
    # Get the y-axis tick positions
    y_ticks = ax.get_yticks()
    
    # Get the first and last tick positions
    y_top = y_ticks[-1]
    y_bottom = y_ticks[0]
    
    total_distance = y_top - y_bottom
    step_per_label = total_distance / (no_y_vals - 1)
    
    # Freeze the current plot limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    cumulative_vals = 0
    for index, label in enumerate(labels):
        y_pos = y_top - step_per_label * (cumulative_vals + vals_per_label[label]/2 - 0.5)
        curr_x_val = x_offset + stagger * (index % 2)
        
        if not "empty" in label and vals_per_label[label] > 0:
            if type(color) == str:
                curr_color = color
            elif type(color) == dict:
                curr_color = color[label]
            else:
                raise Exception("Color should be string or a dict!")
            
            ax.text(
                curr_x_val,
                y_pos,
                label,
                ha='right',
                va='center',
                fontsize=fontsize,
                fontweight=fontweight,
                color=curr_color,
                rotation=rotation,
                transform=ax.transData
            )
            
            adjustment_for_single_value = 0.4 if vals_per_label[label] == 1 else 0
            
            # Draw vertical line
            ax.plot(
                [curr_x_val + 0.2, curr_x_val + 0.2],
                [y_top - step_per_label * (cumulative_vals) + adjustment_for_single_value, 
                y_top - step_per_label * (cumulative_vals + vals_per_label[label] - 1) - adjustment_for_single_value],
                color=curr_color,
                linewidth=linewidth,
                clip_on=False,
                zorder=0
            )
        
        cumulative_vals += vals_per_label[label]
    
    # Restore original plot limits to prevent them from changing
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

def cm_to_px(cm):
    return cm * 27.6

def create_sankey_diagram_strength(
    df,
    column1_name=None, column1_order=None,
    column2_name=None, column2_order=None,
    column3_name=None, column3_order=None,
    height_add=0,
    color1="rgba(31, 119, 180, 0.8)",
    color2="rgba(44, 160, 44, 0.8)",
    color3="rgba(255, 127, 14, 0.8)",
    title=None,
    link_color_df=None,
    color_column="color",
    middle_pos_value=None,
):
    """
    Create a Sankey diagram with preserved node ordering and proper column spacing.
    Optionally color specific connections using `link_color_df`.
    """

    # get order of values in the columns
    column1_values = pd.unique(df[column1_name]).tolist()
    column2_values = pd.unique(df[column2_name]).tolist()
    column3_values = pd.unique(df[column3_name]).tolist()
    
    # Use provided orders or by the order of the values in the dataframe
    if column1_order is not None:
        missing_column1_values = set(df[column1_name].unique()) - set(column1_order)
        if missing_column1_values:
            raise ValueError(f"Column1 order missing values: {missing_column1_values}")
        column1_values = [value for value in column1_order if value in df[column1_name].unique()]
        
    if column2_order is not None:
        missing_column2_values = set(df[column2_name].unique()) - set(column2_order)
        if missing_column2_values:
            raise ValueError(f"Column2 order missing values: {missing_column2_values}")
        column2_values = [value for value in column2_order if value in df[column2_name].unique()]
        
    if column3_order is not None:
        missing_column3_values = set(df[column3_name].unique()) - set(column3_order)
        if missing_column3_values:
            raise ValueError(f"Column3 order missing values: {missing_column3_values}")
        column3_values = [value for value in column3_order if value in df[column3_name].unique()]

    # Create mapping dictionaries with ordered indices
    column1_to_idx = {t: idx for idx, t in enumerate(column1_values)}
    column2_to_idx = {c: idx + len(column1_values) for idx, c in enumerate(column2_values)}
    column3_to_idx = {r: idx + len(column1_values) + len(column2_values) for idx, r in enumerate(column3_values)}
    
    # --- NEW: build explicit edge color map if provided
    edge_color_map = {}
    if link_color_df is not None and len(link_color_df) > 0:
        for _, row in link_color_df.iterrows():
            if color_column not in row or pd.isna(row[color_column]):
                continue
            clr = str(row[color_column])

            # map (column1 -> column2) if both present
            if (column1_name in row and column2_name in row and
                pd.notna(row[column1_name]) and pd.notna(row[column2_name])):
                t = row[column1_name]
                c = row[column2_name]
                if t in column1_to_idx and c in column2_to_idx:
                    edge_color_map[(column1_to_idx[t], column2_to_idx[c])] = clr

            # map (column2 -> column3) if both present
            if (column2_name in row and column3_name in row and
                pd.notna(row[column2_name]) and pd.notna(row[column3_name])):
                c = row[column2_name]
                r = row[column3_name]
                if c in column2_to_idx and r in column3_to_idx:
                    edge_color_map[(column2_to_idx[c], column3_to_idx[r])] = clr

    # Calculate link values while preserving order
    links = []
    link_colors = []

    # default/metadata colors (used only if link_color_df is None)
    meta_to_color = {
        'upregulation': 'rgba(220, 50, 50, 0.45)',
        'upregulated':  'rgba(220, 50, 50, 0.45)',
        'downregulation': 'rgba(50, 100, 220, 0.45)',
        'downregulated':  'rgba(50, 100, 220, 0.45)',
    }
    default_link_color = 'rgba(0,0,0,0.15)'

    # First column to second column (treatments to cell types)
    for t in column1_values:
        for c in column2_values:
            count = len(df[(df[column1_name] == t) & (df[column2_name] == c)])
            if count > 0:
                s = column1_to_idx[t]
                tg = column2_to_idx[c]
                links.append({'source': s, 'target': tg, 'value': count})

                # Use explicit color if provided, else default
                link_colors.append(edge_color_map.get((s, tg), default_link_color))

    # Second column to third column (cell types to responses)
    use_explicit = len(edge_color_map) > 0  # if we have any explicit mapping, we use single aggregated edges
    has_metadata = ('metadata' in df.columns) and not use_explicit

    for c in column2_values:
        for r in column3_values:
            sub = df[(df[column2_name] == c) & (df[column3_name] == r)]
            if len(sub) == 0:
                continue

            s = column2_to_idx[c]
            tg = column3_to_idx[r]

            if use_explicit:
                # aggregate into a single edge and color via explicit map (or default)
                links.append({'source': s, 'target': tg, 'value': int(len(sub))})
                link_colors.append(edge_color_map.get((s, tg), default_link_color))
            elif has_metadata:
                # split by metadata (original behavior)
                vc = sub['metadata'].astype(str).value_counts()
                for meta_value, cnt in vc.items():
                    clr = meta_to_color.get(meta_value.lower(), default_link_color)
                    links.append({'source': s, 'target': tg, 'value': int(cnt)})
                    link_colors.append(clr)
            else:
                # fallback single neutral edge
                links.append({'source': s, 'target': tg, 'value': int(len(sub))})
                link_colors.append(default_link_color)

    # Extract sources, targets, and values from links
    sources = [link['source'] for link in links]
    targets = [link['target'] for link in links]
    values  = [link['value']  for link in links]

    # Create color scheme
    n_column1_values = len(column1_values)
    n_column2_values = len(column2_values)
    n_column3_values = len(column3_values)
    
    colors = (
        [color1] * n_column1_values +
        [color2] * n_column2_values +
        [color3] * n_column3_values
    )
    
    # Create node labels in the correct order
    display_labels = column1_values + column2_values + column3_values
    
    # Calculate node positions
    if middle_pos_value:
        x_positions = [0.0]*n_column1_values + [middle_pos_value]*n_column2_values + [1.0]*n_column3_values
    elif n_column1_values > n_column3_values:
        x_positions = [0.0]*n_column1_values + [0.5]*n_column2_values + [1.0]*n_column3_values
    else:
        x_positions = [0.0]*n_column1_values + [0.3]*n_column2_values + [1.0]*n_column3_values
    
    # Create y positions that spread nodes evenly within their columns
    y_column1_values = np.linspace(0.05, 0.95, n_column1_values) if n_column1_values > 1 else [0.5]
    y_column2_values = np.linspace(0.05, 0.95, n_column2_values) if n_column2_values > 1 else [0.5]
    y_column3_values = np.linspace(0.05, 0.95, n_column3_values) if n_column3_values > 1 else [0.5]
    y_positions = list(y_column1_values) + list(y_column2_values) + list(y_column3_values)
    
    # Create Sankey Diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=5,
            thickness=10,
            line=dict(color='black', width=0.5),
            label=display_labels,
            color=colors,
            x=x_positions,
            y=y_positions,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
            hovertemplate='Source: %{source.label}<br>Target: %{target.label}<br>Value: %{value}<extra></extra>',
            line=dict(color='rgba(0,0,0,0)', width=0)
        ),
        arrangement='snap'
    )])
    
    height = cm_to_px((max(n_column1_values, n_column2_values, n_column3_values)+height_add)/3.3)
    width = cm_to_px(4)
    fig.update_layout(
        height=height,
        width=width,
        margin=dict(l=2, r=2, t=4, b=4),
        showlegend=False,
        font=dict(family="Helsana", size=6, color="black"),
        title=title
    )
    return fig, height, width

def test_pairs(data, x, y, pairs, test=mannwhitneyu, **test_kwargs):
    """Run pairwise tests and return a tidy results DataFrame."""
    results = []

    for a, b in pairs:
        values_a = data.loc[data[x] == a, y].dropna()
        values_b = data.loc[data[x] == b, y].dropna()

        result = test(values_a, values_b, **test_kwargs)

        results.append({
            "group1": a,
            "group2": b,
            "statistic": result.statistic,
            "pvalue": result.pvalue,
            "n1": len(values_a),
            "n2": len(values_b),
        })

    return pd.DataFrame(results)

def annotate_pair_results(
    ax,
    results,
    order,
    p_col="pvalue",
    fmt="{:.3g}",
    margin=0.15,
    h=0.05,
    pad=0.05,
    gap=0.05,
    fontsize=6,
    lw=1,
    color="k",
    loc="inside",
    data_max=None,
    data_min=None,
):
    """Draw brackets using precomputed pairwise-test results."""
    pos = {name: i for i, name in enumerate(order)}

    items = []
    for row_idx, row in results.iterrows():
        a, b = row["group1"], row["group2"]
        i, j = pos[a], pos[b]

        if i < j:
            left_group, right_group = a, b
        else:
            left_group, right_group = b, a

        items.append({
            "left": min(i, j),
            "right": max(i, j),
            "group1": left_group,
            "group2": right_group,
            "label": fmt(row[p_col]) if callable(fmt) else fmt.format(row[p_col]),
            "result_index": row_idx,
        })

    # Widest brackets first, so spanning comparisons are placed higher.
    items.sort(key=lambda item: item["right"] - item["left"], reverse=True)

    rows = []
    placement = []

    for item in items:
        left = item["left"]
        right = item["right"]

        for row_number, occupied in enumerate(rows):
            no_overlap = all(
                right <= occupied_left or left >= occupied_right
                for occupied_left, occupied_right in occupied
            )

            if no_overlap:
                occupied.append((left, right))
                placement.append((item, row_number))
                break
        else:
            rows.append([(left, right)])
            placement.append((item, len(rows) - 1))

    if loc == "outside":
        import matplotlib.transforms as mtransforms

        trans = mtransforms.blended_transform_factory(
            ax.transData,
            ax.transAxes,
        )
        y0 = 1.0 + pad
        step = h + gap
        bracket_height = h

    elif loc == "inside":
        trans = ax.transData

        if data_max is None:
            data_max = ax.get_ylim()[1]
        if data_min is None:
            data_min = ax.get_ylim()[0]

        yrange = data_max - data_min

        if yrange == 0:
            yrange = 1

        y0 = data_max + pad * yrange
        step = (h + gap) * yrange
        bracket_height = h * yrange

    else:
        raise ValueError("loc must be 'inside' or 'outside'")

    fig = ax.figure
    fig.canvas.draw()

    ax_width_px = ax.get_window_extent().width
    x_min, x_max = ax.get_xlim()
    pixels_per_xunit = ax_width_px / (x_max - x_min)

    for item, row_number in placement:
        left = item["left"]
        right = item["right"]
        label = item["label"]

        x1 = left + margin
        x2 = right - margin
        xm = (x1 + x2) / 2

        y_bottom = y0 + row_number * step
        y_top = y_bottom + bracket_height

        label_width = len(label) * fontsize * 0.65 / pixels_per_xunit
        half_label_width = label_width / 2 + 0.03

        ax.plot(
            [x1, x1, xm - half_label_width],
            [y_bottom, y_top, y_top],
            lw=lw,
            color=color,
            transform=trans,
            clip_on=False,
        )
        ax.plot(
            [xm + half_label_width, x2, x2],
            [y_top, y_top, y_bottom],
            lw=lw,
            color=color,
            transform=trans,
            clip_on=False,
        )
        ax.text(
            xm,
            y_top,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            transform=trans,
            clip_on=False,
        )

    if loc == "inside" and rows:
        top = y0 + (len(rows) - 1) * step + bracket_height
        ax.set_ylim(
            top=max(
                ax.get_ylim()[1],
                top + gap * yrange,
            )
        )

def annotate_pvalues(
    ax,
    results,
    order,
    group_col='group2',
    p_col='pvalue',
    fmt='{:.3g}',
    pad=0.05,
    fontsize=6,
    color='k',
):
    """
    Add one p-value above each category, without brackets.

    results must contain:
        group_col: plotted category name
        p_col: p-value
    """
    pos = {name: i for i, name in enumerate(order)}
    fmt_fn = fmt if callable(fmt) else lambda p: fmt.format(p)

    results = results[
        results[group_col].isin(pos)
    ].copy()
    trans = ax.get_xaxis_transform()
    y_text = 1 + pad
    ax.text(-0.5, y_text, "p=", ha='right', va='bottom', transform=trans)
    for _, row in results.iterrows():
        ax.text(
            pos[row[group_col]],
            y_text,
            fmt_fn(row[p_col]) if type(row[p_col]) != str else row[p_col],
            ha='center',
            va='bottom',
            fontsize=fontsize,
            color=color,
            transform=trans,
            clip_on=False,
        )
