"""Candidate-complete secondary-response Sankey using saved attribution and config styles."""

from html import escape
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _symbol(style, white_inner):
    """Translate configured Matplotlib markers to matching label glyphs."""
    marker = style.get("marker", "")
    if "sankey_symbol" in style:
        return style["sankey_symbol"]
    shapes = {"o": ("○", "●"), "D": ("◇", "◆"), "d": ("◇", "◆"), "^": ("△", "▲"),
              "v": ("▽", "▼"), "s": ("□", "■"), "p": ("⬠", "⬟"), "h": ("⬡", "⬢"),
              "H": ("⬡", "⬢"), "<": ("◁", "◀"), ">": ("▷", "▶"), "*": ("☆", "★"),
              "_": ("−", "−"), "x": ("×", "×"), "X": ("×", "×"), ".": ("·", "·"),
              "+": ("+", "+"), "P": ("✚", "✚"), "|": ("|", "|")}
    if marker in shapes:
        return shapes[marker][0 if marker in white_inner else 1]
    if isinstance(marker, str) and marker.startswith("$") and marker.endswith("$"):
        return marker[1:-1].replace(r"\bowtie", "⋈")
    if marker in {None, "", "none", "None", " "}:
        return ""
    raise ValueError(f"Add sankey_symbol to the config style for marker {marker!r}.")


def _white_inner_set(config):
    """Read the current style setting, with compatibility for older notebooks."""
    return getattr(config, "white_inner_set", getattr(config, "whitter_inner_set", set()))


def _compact_flow_figure(query_cyto, pairs, paths, original_nodes, config, mode, label_formatter, figsize_cm, font_size,
                         title, max_producers=4, max_recipients=3):
    """Contract route stages and group minor cells; preserve their values and expanded audit."""
    fmt = lambda value: escape(str(label_formatter(str(value))))
    def cell_label(cell):
        replacements = {"CD4 T cell": "CD4 T", "CD8 T cell": "CD8 T", "NK CD56low": "NK low", "NK CD56hi": "NK hi",
                        "CD14 Mono": "CD14", "CD16 Mono": "CD16", "B cell": "B"}
        return escape(replacements.get(str(cell), str(cell)))

    producers = paths.loc[paths.target.str.startswith('product::')].copy()
    recipients = paths.loc[paths.source.str.startswith('product::')].copy()
    # Fold the upstream route into the producer's label/hover, not an implied direct edge.
    descriptors = {}
    for source in producers.source.unique():
        node = original_nodes[source]
        cell = node['cell_type']
        if source.startswith('direct::'):
            label, kind = cell_label(cell), 'direct'
        elif source.startswith('relay::'):
            mediator = source.split('::')[-2]
            label, kind = f"{cell_label(cell)}<br>via {fmt(mediator)}", 'relay'
        elif source.startswith('indirect::'):
            label, kind = f"? {cell_label(cell)}", 'unresolved'
        else:
            label, kind = 'Unassigned', 'unresolved'
        descriptors[source] = dict(label=label, kind=kind, detail=node['detail'])

    def grouping(series, maximum):
        totals = series.groupby(series.index).sum().sort_values(ascending=False)
        if len(totals) <= maximum:
            return {key:key for key in totals.index}
        retained = set(totals.head(maximum - 1).index)
        return {key:key if key in retained else 'other' for key in totals.index}

    producer_map = grouping(producers.set_index('source').value, max_producers)
    recipient_map = grouping(recipients.set_index('target').value, max_recipients)
    compact_nodes, compact_edges = {}, []
    def node(key, label, stage, color, detail=''):
        compact_nodes[key] = dict(node_id=key, label=label, stage=stage, color=color, detail=detail)
        return key
    def add(source, target, value, row, segment, color, detail):
        compact_edges.append(dict(source=source, target=target, value=float(value), query_cyto=query_cyto,
            secondary=row.secondary, causal_class=row.causal_class, confidence=row.confidence, direction=row.direction,
            priority=row.priority, segment=segment, color=color, detail=detail, intermediate=row.intermediate))
    node('input', fmt(query_cyto), 0, '#343A40')
    for key, group in producers.groupby(producers.source.map(producer_map), sort=False):
        originals = list(group.source.unique())
        kinds = {descriptors[source]['kind'] for source in originals}
        unresolved = 'unresolved' in kinds
        label = descriptors[originals[0]]['label'] if key != 'other' else ('? Other' if unresolved else 'Other')
        detail = '<br>'.join(original_nodes[source]['label'] + ': ' + descriptors[source]['detail'] for source in originals)
        pid = node('producer::'+key, label, 1, '#BB913E' if unresolved else '#8660A0' if 'relay' in kinds else '#7F909C', detail)
        for _, row in group.iterrows():
            kind = descriptors[row.source]['kind']
            color = 'rgba(190,140,55,.48)' if kind == 'unresolved' else 'rgba(130,95,155,.36)' if kind == 'relay' else 'rgba(100,110,120,.30)'
            full_detail = original_nodes[row.source]['label'] + '<br>' + row.detail
            if kind == 'relay':
                upstream = paths.loc[paths.secondary.eq(row.secondary) & paths.intermediate.eq(row.intermediate)
                    & paths.segment.eq('input-compatible intermediate producer'), 'target'].unique()
                full_detail += '<br>Intermediate producers: ' + ', '.join(original_nodes[key]['label'] for key in upstream)
            add('input', pid, row.value, row, 'producer route (contracted)', color, full_detail)
            add(pid, row.target, row.value, row, row.segment, row.color, full_detail)
    for _, row in pairs.iterrows():
        source = original_nodes['product::'+row.secondary]
        node(source['node_id'], source['label'], 2, source['color'], source['detail'])
    for key, group in recipients.groupby(recipients.target.map(recipient_map), sort=False):
        originals = list(group.target.unique())
        if key == 'other':
            label = 'Other'
        else:
            cell = original_nodes[key]['cell_type']
            label = cell_label(cell) if cell is not None else 'No support' if 'No supported' in original_nodes[key]['detail'] else '? Recipient'
            if key.startswith('recipient::missing::'):
                label = '? ' + label
        detail = '<br>'.join(original_nodes[source]['label'] + ': ' + original_nodes[source]['detail'] for source in originals)
        recipient_color = original_nodes[originals[0]]['color'] if len(originals) == 1 else '#8AA6A2'
        rid = node('recipient::'+key, label, 3, recipient_color, detail)
        for _, row in group.iterrows():
            add(row.source, rid, row.value, row, row.segment, row.color, original_nodes[row.target]['label'] + '<br>' + row.detail)
    table = pd.DataFrame(compact_nodes.values())
    # Cytokines follow selected-pair order. Other nodes follow flow magnitude to reduce crossings.
    compact_paths = pd.DataFrame(compact_edges)
    weights = compact_paths.groupby('target').value.sum()
    table['order'] = -table.node_id.map(weights).fillna(0)
    table = table.sort_values(['stage','order','label'], kind='stable').reset_index(drop=True)
    table['x'] = table.stage.map({0:.005,1:.29,2:.65,3:.995})
    table['y'] = .5
    for _, indices in table.groupby('stage').groups.items():
        table.loc[indices,'y'] = [.5] if len(indices)==1 else np.linspace(.08,.92,len(indices))
    indices = dict(zip(table.node_id,table.index))
    compact_paths['source_index'] = compact_paths.source.map(indices)
    compact_paths['target_index'] = compact_paths.target.map(indices)
    custom = compact_paths[['secondary','segment','causal_class','detail']].to_numpy()
    fig = go.Figure(go.Sankey(arrangement='snap', node=dict(label=table.label, color=table.color, x=table.x,y=table.y,
        customdata=table.detail,pad=3,thickness=3,line=dict(color='#444',width=.25),
        hovertemplate='%{label}<br>%{customdata}<extra></extra>'),
        link=dict(source=compact_paths.source_index,target=compact_paths.target_index,value=compact_paths.value,color=compact_paths.color,
        customdata=custom,hovertemplate='%{source.label} → %{target.label}<br>Final: %{customdata[0]}<br>%{customdata[1]}<br>%{customdata[2]}<br>%{customdata[3]}<br>Display share: %{value:.3f}<extra></extra>')))
    if figsize_cm is None:
        figsize_cm = (6,max(2.5,.42*int(table.groupby('stage').size().max())))
    fig.update_layout(width=int(figsize_cm[0]/2.54*96),height=int(figsize_cm[1]/2.54*96),
        margin=dict(l=2,r=2,t=12 if title else 2,b=2),font=dict(family='Arial',size=font_size,color='#202020'),paper_bgcolor='white',
        title=dict(text=title,x=.01,font=dict(size=font_size)) if title else None,
        meta=dict(mode=mode,compact=True,candidate_count=len(pairs),width_basis='one unit per candidate; not molecular flux',
            nodes=table.to_dict('records'),expanded_nodes=list(original_nodes.values()),expanded_paths=paths.to_dict('records'),
            producer_groups=producer_map,recipient_groups=recipient_map))
    compact_paths.attrs['expanded_paths'] = paths
    return fig, pairs, compact_paths


def _prune_publication_paths(paths, producer_cutoff, relay_cutoff, recipient_cutoff):
    """Prune actual cell paths per candidate, without pooling or renormalizing omitted cells."""
    kept, coverage = [], []
    def add(row, value=None):
        record = row.to_dict()
        if value is not None:
            record['value'] = float(value)
        if record['value'] > 1e-12:
            kept.append(record)
    for product, group in paths.groupby('secondary', sort=False):
        terminal = group.loc[group.target.eq('product::'+product)]
        cell_path = terminal.source.str.startswith(('direct::','relay::','indirect::'))
        thresholds = np.where(terminal.source.str.startswith('direct::'), producer_cutoff, relay_cutoff)
        selected = terminal.loc[~cell_path | terminal.value.ge(thresholds)]
        if selected.empty:
            # Keep an explicit visual accounting node when the caller's cutoff
            # excludes every assigned producer path.
            summary = terminal.iloc[0].copy()
            summary['source'], summary['target'], summary['value'] = 'input', f'producer-filter::{product}', 1.
            summary['segment'], summary['color'] = 'producer paths below cutoff', 'rgba(190,140,55,.46)'
            summary['detail'] = 'Every assigned producer path is below the supplied producer / relay cutoff.'
            add(summary)
            summary['source'], summary['target'] = f'producer-filter::{product}', 'product::' + product
            summary['segment'] = 'observed change after producer filtering'
            add(summary)
            coverage.append(dict(secondary=product, production_retained=0., production_omitted=1.,
                                 recipient_fraction_retained=0.))
            continue
        incoming_total = 0.
        # Direct and unresolved paths terminate in known producers or explicit attribution gaps.
        for _, row in selected.loc[~selected.source.str.startswith('relay::')].iterrows():
            add(row)
            incoming_total += row.value
            predecessors = group.loc[group.target.eq(row.source)]
            if row.source.startswith('direct::'):
                predecessors = predecessors.loc[predecessors.segment.eq('input-compatible producer')]
            for _, pred in predecessors.iterrows():
                add(pred, row.value)
                if pred.source != 'input':
                    upstream = group.loc[group.source.eq('input') & group.target.eq(pred.source)]
                    for _, edge in upstream.iterrows():
                        add(edge, row.value)
        # Keep complete relay paths; propagate pruning through both producer stages.
        for intermediate, selected_relay in selected.loc[selected.source.str.startswith('relay::')].groupby('intermediate'):
            relay = group.loc[group.segment.eq('relay attributed change') & group.intermediate.eq(intermediate)]
            total = relay.value.sum()
            primary = group.loc[group.segment.eq('intermediate attributed change') & group.intermediate.eq(intermediate)]
            primary = primary.loc[primary.value.ge(producer_cutoff)]
            fraction_primary = primary.value.sum() / total if total else 0
            fraction_final = selected_relay.value.sum() / total if total else 0
            for _, row in primary.iterrows():
                value = row.value * fraction_final
                add(row, value)
                for _, upstream in group.loc[group.source.eq('input') & group.target.eq(row.source)
                        & group.segment.eq('input-compatible intermediate producer') & group.intermediate.eq(intermediate)].iterrows():
                    add(upstream, value)
            for _, row in selected_relay.iterrows():
                value = row.value * fraction_primary
                add(row, value)
                incoming_total += value
                for _, upstream in group.loc[group.target.eq(row.source) & group.segment.eq('accepted intermediate route')].iterrows():
                    add(upstream, value)
        if incoming_total <= 0:
            raise ValueError(f'No complete producer paths remain for {product}; lower the producer cutoffs.')
        response = group.loc[group.source.eq('product::'+product)]
        response_keep = response.loc[
            response.raw_similarity.ge(recipient_cutoff)
            | response.target.str.startswith('recipient-gap::')
        ]
        response = response_keep
        if response.empty and not group.loc[group.source.eq('product::'+product)].empty:
            summary = group.loc[group.source.eq('product::'+product)].iloc[0].copy()
            summary['target'], summary['value'] = f'recipient-filter::{product}', 1.
            summary['segment'], summary['color'] = 'recipient similarity below cutoff', 'rgba(180,180,180,.38)'
            summary['detail'] = 'Every supported recipient is below the supplied raw-similarity cutoff.'
            add(summary, incoming_total)
        for _, row in response.iterrows():
            add(row, row.value * incoming_total)
        coverage.append(dict(secondary=product, production_retained=incoming_total,
            production_omitted=1-incoming_total, recipient_fraction_retained=float(response.value.sum())))
    result = pd.DataFrame(kept).drop(columns=['source_index','target_index'], errors='ignore')
    return result, coverage


def _place_publication_labels(table, width, height, font_size, thickness):
    """Jointly place labels using reversible choices, alternate wraps and the full panel height."""
    import re
    from html import unescape
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    font = FontProperties(family='Arial')
    def measure(label):
        lines = [unescape(re.sub('<[^>]*>', '', line)) for line in label.split('<br>')]
        widths = [TextPath((0, 0), line, size=font_size, prop=font).get_extents().width if line else 0 for line in lines]
        return max(widths) + 4, len(lines) * font_size * 1.15 + 2
    def intersects(a, b):
        return a[0] < b[2] + 1 and a[2] + 1 > b[0] and a[1] < b[3] + 1 and a[3] + 1 > b[1]
    domains = {}
    for idx, node in table.iterrows():
        nx, ny = node.x * width, node.y * height
        preferred = 'right' if node.stage <= 2 else 'left'
        variants = list(dict.fromkeys([node.short_label, node.short_label.replace('<br>', ' ')]))
        candidates = []
        for variant_index, label in enumerate(variants):
            tw, th = measure(label)
            if tw > width or th > height:
                continue
            # Recipient labels are deliberately fixed at the node centre.  A
            # displaced right-hand label makes its associated cell ambiguous.
            # If centred recipient labels cannot fit, the cutoff should be
            # raised rather than silently moving the annotation.
            desired = np.clip(ny - th / 2, 0, height - th)
            if node.stage == 5:
                y_positions, sides = [float(desired)], [preferred]
            else:
                y_positions = sorted(
                    set([float(desired), 0., height - th] + list(np.arange(0, height - th, font_size / 2))),
                    key=lambda y: abs(y - desired),
                )
                sides = [preferred, 'left' if preferred == 'right' else 'right']
            for side in sides:
                x = nx + thickness / 2 + 2 if side == 'right' else nx - thickness / 2 - 2 - tw
                if x < 0 or x + tw > width:
                    continue
                for y in y_positions:
                    box = (float(x), float(y), float(x + tw), float(y + th))
                    cost = abs(y + th / 2 - ny) + (0 if side == preferred else 2) + variant_index
                    candidates.append((float(cost), box, side, label))
        domains[idx] = sorted(candidates, key=lambda candidate: candidate[0])
        if not candidates:
            raise ValueError(f'Label {unescape(node.short_label)!r} cannot fit inside the figure at font_size={font_size:g}.')
    # Minimum-remaining-values selection avoids reserving a flexible label's position first.
    # Forward checking rejects choices that leave any remaining label without a position.
    visited = 0
    limit = 12000
    def search(remaining, placed):
        nonlocal visited
        visited += 1
        if not remaining:
            return placed
        if visited > limit:
            return None
        idx = min(remaining, key=lambda key: (len(remaining[key]), remaining[key][0][0]))
        for candidate in remaining[idx]:
            reduced = {key: [choice for choice in choices if not intersects(candidate[1], choice[1])]
                       for key, choices in remaining.items() if key != idx}
            if any(not choices for choices in reduced.values()):
                continue
            result = search(reduced, {**placed, idx: candidate})
            if result is not None:
                return result
            if visited > limit:
                break
        return None
    solution = search(domains, {})
    if solution is None:
        raise ValueError('Joint label placement did not find a non-overlapping layout. Adjust stage_x, raise a stage cutoff, or increase figure height; font size was preserved.')
    return solution, visited


def _publication_flow_figure_once(query_cyto, pairs, paths, original_nodes, mode, figsize_cm, font_size, title,
                                  producer_cutoff, relay_cutoff, recipient_cutoff, stage_x):
    """Explicit route stages, 6 pt type, measured label placement and no pooled cells."""
    import re
    from html import unescape
    displayed, coverage = _prune_publication_paths(paths, producer_cutoff, relay_cutoff, recipient_cutoff)
    # A directly produced selected cytokine can also serve as an intermediate: show it once.
    # Only merge when its retained inputs are in the first producer stage, preserving a forward DAG.
    original_nodes = {key:dict(value) for key,value in original_nodes.items()}
    for key in set(displayed.source) | set(displayed.target):
        if key.startswith('producer-filter::') and key not in original_nodes:
            original_nodes[key] = dict(node_id=key, label='Producers below<br>cutoff', stage=3, color='#BB913E',
                detail='Every assigned producer path is below the supplied cutoff.', cell_type=None)
        elif key.startswith('recipient-filter::') and key not in original_nodes:
            original_nodes[key] = dict(node_id=key, label='Recipients below<br>cutoff', stage=5, color='#B4B4B4',
                detail='Every supported recipient is below the supplied raw-similarity cutoff.', cell_type=None)
    merged_intermediates = {}
    for key in list(original_nodes):
        if not key.startswith('intermediate::'):
            continue
        target = 'product::' + key.split('::',1)[1]
        if target not in original_nodes:
            continue
        incoming = displayed.loc[displayed.target.eq(target),'source']
        if not incoming.empty and all(original_nodes[source]['stage'] <= 1 for source in incoming):
            displayed['source'] = displayed.source.replace({key:target})
            displayed['target'] = displayed.target.replace({key:target})
            original_nodes[target]['stage'] = 2
            original_nodes[target]['detail'] += '<br>Also an accepted intermediate for another displayed candidate.'
            merged_intermediates[key] = target
    used = set(displayed.source) | set(displayed.target)
    table = pd.DataFrame([dict(original_nodes[key]) for key in original_nodes if key in used])
    weights = pd.concat([displayed.groupby('source').value.sum(), displayed.groupby('target').value.sum()],axis=1).fillna(0).max(axis=1)
    table['flow'] = table.node_id.map(weights)
    table = table.sort_values(['stage','flow','label'], ascending=[True,False,True],kind='stable').reset_index(drop=True)
    stages = sorted(int(stage) for stage in table.stage.unique())
    # Order each column by the weighted positions of its connected neighbours.
    # This keeps an accepted relay visually between its intermediate and final
    # cytokine, rather than sorting columns independently by ribbon width.
    stage_by_node = table.set_index('node_id').stage.to_dict()
    stage_order = {stage: table.loc[table.stage.eq(stage), 'node_id'].tolist() for stage in stages}
    initial_rank = {node: rank for stage in stages for rank, node in enumerate(stage_order[stage])}
    for sweep in range(4):
        ordered_stages = stages[1:] if sweep % 2 == 0 else stages[-2::-1]
        rank = {node: position for stage in stages for position, node in enumerate(stage_order[stage])}
        for stage in ordered_stages:
            scores = {}
            for node_id in stage_order[stage]:
                incoming = displayed.loc[displayed.target.eq(node_id), ['source', 'value']]
                outgoing = displayed.loc[displayed.source.eq(node_id), ['target', 'value']]
                neighbours = pd.concat([
                    incoming.rename(columns={'source': 'node'}), outgoing.rename(columns={'target': 'node'})
                ], ignore_index=True)
                neighbours = neighbours.loc[neighbours.node.map(stage_by_node).ne(stage)]
                if neighbours.empty:
                    scores[node_id] = float(initial_rank[node_id])
                else:
                    scores[node_id] = float(np.average(neighbours.node.map(rank), weights=neighbours.value))
            stage_order[stage].sort(key=lambda node_id: (scores[node_id], initial_rank[node_id]))
    ordered_ids = [node_id for stage in stages for node_id in stage_order[stage]]
    table = table.set_index('node_id').loc[ordered_ids].reset_index()
    # Give producer labels and the last two columns more room than the short input label.
    defaults = {2:[.015,.985],3:[.015,.30,.985],4:[.015,.18,.46,.985],
                5:[.015,.13,.31,.56,.985],6:[.015,.11,.25,.39,.58,.985]}
    positions = dict(zip(stages,defaults[len(stages)]))
    if stage_x is not None:
        unknown = set(stage_x) - set(range(6))
        if unknown:
            raise ValueError(f'Unknown stage_x keys: {sorted(unknown)}; stages run from 0 to 5.')
        positions.update({stage:float(x) for stage,x in stage_x.items() if stage in stages})
        if any(not 0 < positions[stage] < 1 for stage in stages) or any(positions[a] >= positions[b] for a,b in zip(stages,stages[1:])):
            raise ValueError('stage_x must increase strictly across stages, with positions between 0 and 1.')
    if figsize_cm is None:
        figsize_cm = (6,2.5)
    width,height = int(figsize_cm[0]/2.54*96),int(figsize_cm[1]/2.54*96)
    margin=dict(l=3,r=3,t=14 if title else 3,b=3)
    w,h=width-margin['l']-margin['r'],height-margin['t']-margin['b']
    if min(w,h) <= 0:
        raise ValueError('Figure dimensions are too small for the requested margins.')
    pad,thickness=3.,2.5
    scale=min((h-pad*(len(group)-1))/group.flow.sum() for _,group in table.groupby('stage'))
    if scale <= 0:
        raise ValueError('Too many nodes for this height; increase cutoffs or figure height.')
    table['x']=table.stage.map(positions)
    table['y']=.5
    # Spread each column over the available height instead of centering small columns in whitespace.
    for _,indices in table.groupby('stage').groups.items():
        heights=table.loc[indices,'flow'].to_numpy()*scale
        gap=(h-heights.sum())/(len(indices)-1) if len(indices)>1 else 0
        cursor=0 if len(indices)>1 else (h-heights.sum())/2
        for idx,node_height in zip(indices,heights):
            table.loc[idx,'y']=(cursor+node_height/2)/h
            cursor+=node_height+gap
    labels=[]
    for _,node in table.iterrows():
        cell=node.get('cell_type')
        if node.stage == 5:
            label = re.sub(r'<[^>]+>', ' ', node.label)
            label = ' '.join(label.split())
        elif pd.notna(cell):
            label=str(cell).replace(' T cell',' T').replace('B cell','B')
            label=label.replace('NK CD56low','NK CD56lo').replace('NK CD56hi','NK CD56hi')
        else:
            label=re.sub(r'<[^>]+>', ' ', node.label)
            label=' '.join(label.split())
            label=label.replace('Intermediate unidentified','? relay').replace('Producer / receptor attribution unknown','? producer')
            label=label.replace('Direct producer unassigned','? producer').replace('Indirect producer unassigned','? producer')
            label=label.replace('No supported recipient','No support').replace('Recipient evidence unavailable','? response')
        labels.append(label)
    table['short_label']=labels
    # Joint placement can revisit earlier choices, unwrap labels and use the entire height.
    # When the requested canvas physically cannot hold every label, preserve the
    # complete Sankey and expose names/details through hover rather than failing.
    try:
        placements, search_steps = _place_publication_labels(table,w,h,font_size,thickness)
        label_placement_status = 'placed'
    except ValueError:
        placements, search_steps, label_placement_status = {}, 0, 'hover-only: insufficient label space'
    boxes,annotations,leaders,recipient_outlines=[],[],[],[]
    for idx,node in table.iterrows():
        if idx not in placements:
            role = node.get('recipient_role')
            if node.stage == 5 and pd.notna(role):
                dash = {'facultative': 'dot', 'obligate': 'solid', 'unknown': 'dash'}[role]
                color = {'facultative': '#6F7E86', 'obligate': '#238178', 'unknown': '#A97E31'}[role]
                half_height = max(3., node.flow * scale / 2)
                recipient_outlines.append(dict(type='rect',xref='paper',yref='paper',
                    x0=(node.x * w - thickness / 2 - 1) / w, x1=(node.x * w + thickness / 2 + 1) / w,
                    y0=1 - (node.y * h + half_height) / h, y1=1 - (node.y * h - half_height) / h,
                    line=dict(color=color,width=.8,dash=dash), fillcolor='rgba(0,0,0,0)'))
            continue
        _,box,side,label=placements[idx]
        boxes.append(box)
        x0,y0,x1,y1=box
        annotations.append(dict(x=x0/w,y=1-y0/h,xref='paper',yref='paper',xanchor='left',yanchor='top',text=label,
            showarrow=False,align='left',borderpad=1,bgcolor='rgba(255,255,255,.75)',font=dict(family='Arial',size=font_size,color='#222222')))
        role = node.get('recipient_role')
        if node.stage == 5 and pd.notna(role):
            dash = {'facultative': 'dot', 'obligate': 'solid', 'unknown': 'dash'}[role]
            color = {'facultative': '#6F7E86', 'obligate': '#238178', 'unknown': '#A97E31'}[role]
            recipient_outlines.append(dict(type='rect',xref='paper',yref='paper',x0=x0/w,y0=1-y1/h,x1=x1/w,y1=1-y0/h,
                line=dict(color=color,width=.8,dash=dash),fillcolor='rgba(0,0,0,0)'))
        if abs((y0+y1)/2-node.y*h)>font_size/2:
            leaders.append(dict(type='line',xref='paper',yref='paper',x0=node.x,y0=1-node.y,
                x1=(x0 if side=='right' else x1)/w,y1=1-(y0+y1)/2/h,line=dict(color='#666',width=.4)))
    index=dict(zip(table.node_id,table.index))
    displayed['source_index']=displayed.source.map(index)
    displayed['target_index']=displayed.target.map(index)
    node_labels = [''] * len(table) if label_placement_status == 'placed' else table.short_label.tolist()
    fig=go.Figure(go.Sankey(arrangement='fixed',node=dict(label=node_labels,x=table.x,y=table.y,color=table.color,
        customdata=np.column_stack([table.label,table.detail]),pad=pad,thickness=thickness,line=dict(color='#444',width=.25),
        hovertemplate='%{customdata[0]}<extra></extra>'),
        link=dict(source=displayed.source_index,target=displayed.target_index,value=displayed.value,color=displayed.color,
        customdata=displayed[['secondary','segment','detail']].to_numpy(),
        hovertemplate='%{customdata[0]}: %{customdata[1]}<extra></extra>')))
    fig.update_layout(width=width,height=height,margin=margin,annotations=annotations,shapes=leaders + recipient_outlines,
        font=dict(family='Arial',size=font_size,color='#222222'),paper_bgcolor='white',title=dict(text=title,font=dict(size=font_size)) if title else None,
        meta=dict(mode=mode,layout='publication',font_size_pt=font_size,nodes=table.to_dict('records'),label_boxes=boxes,label_search_steps=search_steps,
            label_placement_status=label_placement_status,
            coverage=coverage,stage_x=positions,merged_intermediates=merged_intermediates,expanded_nodes=list(original_nodes.values()),expanded_paths=paths.to_dict('records'),
            cutoffs=dict(producer=producer_cutoff,relay=relay_cutoff,recipient_similarity=recipient_cutoff)))
    displayed.attrs['expanded_paths']=paths
    displayed.attrs['coverage']=pd.DataFrame(coverage)
    return fig,pairs,displayed


def _publication_flow_figure(query_cyto, pairs, paths, original_nodes, mode, figsize_cm, font_size, title,
                             producer_cutoff, relay_cutoff, recipient_cutoff, stage_x):
    """Build a publication Sankey using exactly the caller-supplied cutoffs."""
    fig, selected, displayed = _publication_flow_figure_once(
        query_cyto, pairs, paths, original_nodes, mode, figsize_cm, font_size, title,
        producer_cutoff, relay_cutoff, recipient_cutoff, stage_x)
    fig.layout.meta['requested_cutoffs'] = dict(
        producer=producer_cutoff, relay=relay_cutoff, recipient_similarity=recipient_cutoff)
    return fig, selected, displayed


def plot_secondary_flow_sankey(query_cyto, *, pair_table, producer_shares_by_pair, producer_details_by_pair,
        support_by_pair, similarity_by_pair, config, mode="all", direction=None, secondary=None, downstream_cytokines=None, n_pairs=None,
        recipient_role="all", similarity_reference=.8,
        cell_type_order=None, label_formatter=str, figsize_cm=None, font_size=None, title=None,
        compact=True, max_producers=4, max_recipients=3, layout=None,
        minimum_producer_share=.10, minimum_relay_share=.10, minimum_recipient_similarity=None,
        minimum_recipient_share=None, stage_x=None):
    """Show all candidate production/response paths, including unassigned fractions.

    Default layout='publication' keeps explicit route stages and prunes by per-candidate shares,
    without grouping cell types. font_size is passed directly to Plotly (default 6). stage_x maps stage 0..5 to
    increasing positions in (0,1). Producer and relay cutoffs apply to route shares;
    minimum_recipient_similarity applies to raw direction-aligned response similarity before
    the retained values are normalized into ribbon widths. minimum_recipient_share is retained
    as a compatibility alias for that argument.
    Omitted shares are not pooled or renormalized; paths.attrs['coverage'] reports retained evidence.
    layout='tiny' retains the earlier contracted/grouped view; compact=False selects the expanded audit.
    Compact paths.attrs['expanded_paths'] and figure.layout.meta retain the original routes.
    mode='all' includes every candidate for the input, irrespective of class or priority.
    mode='supported' uses config.consistent_confidence. Optional n_pairs limits either mode.
    downstream_cytokines restricts final products, not intermediate route evidence.
    recipient_role filters response cells: 'obligate' have low input-receptor
    expression, 'facultative' also permit a direct input effect, and 'unknown'
    lack input-receptor coverage.
    similarity_reference controls the response-ribbon saturation; r at or above
    this value receives full saturation (default 0.8).
    figsize_cm is (width, height) in centimetres; None sizes the figure automatically.
    Each candidate contributes one unit: saved median D/I shares split production; donor-mean
    shares allocate cell types within D/I. Accepted overlap allocates the identified relay;
    its remainder stays unresolved. Recipient widths use normalized aligned similarity.
    These are evidence allocations, not molecular flux or estimates of causal mediation.

    Returns (figure, displayed_pairs, path_table). Full node metadata is in figure.layout.meta.
    """
    layout = layout or ('publication' if compact else 'expanded')
    if layout not in {'publication','tiny','expanded'}:
        raise ValueError("layout must be 'publication', 'tiny', or 'expanded'.")
    if font_size is None:
        font_size = 6
    font_size = float(font_size)
    if not np.isfinite(font_size) or font_size <= 0:
        raise ValueError('font_size must be positive.')
    if minimum_recipient_similarity is not None and minimum_recipient_share is not None:
        raise ValueError('Use minimum_recipient_similarity, not both recipient cutoff arguments.')
    if minimum_recipient_similarity is None:
        minimum_recipient_similarity = .15 if minimum_recipient_share is None else minimum_recipient_share
    if any(not np.isfinite(v) or not 0 <= v <= 1 for v in
           (minimum_producer_share, minimum_relay_share, minimum_recipient_similarity)):
        raise ValueError('Stage cutoffs must be finite fractions between 0 and 1.')
    if any(not isinstance(n, int) or n < 2 for n in (max_producers, max_recipients)):
        raise ValueError("max_producers and max_recipients must be integers >= 2.")
    if mode not in {"all", "supported"}:
        raise ValueError("mode must be 'all' or 'supported'.")
    if recipient_role not in {'all', 'obligate', 'facultative', 'unknown'}:
        raise ValueError("recipient_role must be 'all', 'obligate', 'facultative', or 'unknown'.")
    if not np.isfinite(similarity_reference) or similarity_reference <= 0:
        raise ValueError('similarity_reference must be a positive finite value.')
    required = {"query_cyto", "secondary", "direction", "priority", "confidence", "causal_class",
                "direct_producer_share", "indirect_producer_share", "best_intermediate"}
    missing = required - set(pair_table.columns)
    if missing:
        raise ValueError(f"pair_table is missing {sorted(missing)}")
    if pair_table.duplicated(["query_cyto", "secondary"]).any():
        raise ValueError("Expected one row per input/downstream cytokine pair.")
    pairs = pair_table.loc[pair_table.query_cyto.eq(query_cyto)].copy()
    if mode == "supported":
        pairs = pairs.loc[pairs.confidence.isin(config.consistent_confidence)]
    if direction is not None:
        pairs = pairs.loc[pairs.direction.eq(direction)]
    if secondary is not None and downstream_cytokines is not None:
        raise ValueError("Use secondary or downstream_cytokines, not both.")
    if secondary is not None:
        pairs = pairs.loc[pairs.secondary.eq(secondary)]
    if downstream_cytokines is not None:
        selected_products = [downstream_cytokines] if isinstance(downstream_cytokines, str) else list(downstream_cytokines)
        pairs = pairs.loc[pairs.secondary.isin(selected_products)]
    if figsize_cm is not None:
        try:
            dimensions = np.asarray(figsize_cm, dtype=float)
            valid_size = dimensions.shape == (2,) and np.isfinite(dimensions).all() and (dimensions > 0).all()
        except (TypeError, ValueError):
            valid_size = False
        if not valid_size:
            raise ValueError("figsize_cm must be two finite positive values: (width, height) in centimetres.")
        figsize_cm = tuple(dimensions)
    pairs = pairs.sort_values(["priority", "secondary"], ascending=[False, True], na_position="last")
    if n_pairs is not None:
        if not isinstance(n_pairs, int) or n_pairs < 1:
            raise ValueError("n_pairs must be a positive integer or None.")
        pairs = pairs.head(n_pairs)
    if pairs.empty:
        raise ValueError(f"No {mode} candidates for {query_cyto!r} under these filters.")
    lookup = {(r.query_cyto, r.secondary): r for _, r in pair_table.iterrows()}
    signs = {"secondary increased": 1, "secondary reduced": -1}
    colors = {1: "rgba(185,65,65,.40)", -1: "rgba(65,105,180,.40)"}
    neutral, amber, teal = "rgba(100,110,120,.30)", "rgba(190,140,55,.46)", "rgba(35,135,125,.36)"
    nodes, edges = {}, []
    order = {cell: idx for idx, cell in enumerate(cell_type_order or [])}
    fmt = lambda value: escape(str(label_formatter(str(value))))

    def production_color(direction, contribution):
        """Make the exact cell contribution visible even when its ribbon is thin."""
        strength = min(max(float(contribution), 0.), 1.)
        hue = 4 if direction > 0 else 216
        return f"hsl({hue}, {18 + 72 * strength:.0f}%, {80 - 36 * strength:.0f}%)"

    def unresolved_color(contribution):
        strength = min(max(float(contribution), 0.), 1.)
        return f"hsl(39, {22 + 58 * strength:.0f}%, {84 - 34 * strength:.0f}%)"

    def node(key, label, stage, color, detail="", cell=None, recipient_role=None):
        if key not in nodes:
            nodes[key] = dict(node_id=key, label=label, stage=stage, color=color, detail=detail,
                              cell_type=cell, recipient_role=recipient_role)
        return key

    def edge(source, target, value, row, segment, color, detail="", intermediate=None, raw_similarity=np.nan):
        if value <= 0:
            return
        edges.append(dict(source=source, target=target, value=float(value), query_cyto=query_cyto,
            secondary=row.secondary, causal_class=row.causal_class, confidence=row.confidence,
            direction=row.direction, priority=row.priority, segment=segment, detail=detail,
            intermediate=intermediate, color=color, raw_similarity=raw_similarity))

    def shares_and_classes(key):
        shares = producer_shares_by_pair.get(key)
        if isinstance(shares, pd.DataFrame):
            shares = shares.mean(axis=0)  # saved retained donors only
        elif shares is None:
            shares = pd.Series(dtype=float)
        shares = pd.Series(shares, dtype=float).replace([np.inf, -np.inf], np.nan).dropna().clip(lower=0)
        details = producer_details_by_pair.get(key, {})
        classes = pd.Series(details.get("producer_receptor_classes", {}), dtype=object).reindex(shares.index).fillna("missing")
        return shares, classes

    def normalize(values):
        values = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna().clip(lower=0)
        values = values[values.gt(0)]
        return values / values.sum() if not values.empty else values

    query_id = node("input", fmt(query_cyto), 0, "#343A40", "Input perturbation; one display unit per downstream candidate.")
    for _, row in pairs.iterrows():
        if row.direction not in signs:
            raise ValueError(f"Unknown direction {row.direction!r}")
        sign, product = signs[row.direction], row.secondary
        key = (query_cyto, product)
        style = config.causal_styles.get(row.causal_class, {})
        glyph = _symbol(style, _white_inner_set(config))
        style_label = style.get("label", row.causal_class)
        product_id = node(f"product::{product}", f"{glyph} {'↑' if sign > 0 else '↓'} {fmt(product)}".strip(), 4,
            "#B94141" if sign > 0 else "#4169AD", f"{escape(style_label)}<br>{escape(str(row.confidence))}<br>Priority: {row.priority}")
        shares, classes = shares_and_classes(key)
        d, i = row.direct_producer_share, row.indirect_producer_share
        attribution_known = pd.notna(d) and pd.notna(i) and np.isfinite(d + i) and min(d, i) >= 0 and np.isclose(d + i, 1)
        if attribution_known:
            d, i = float(d), float(i)
        else:
            d, i = 0., 0.

        # Directly attributable production remains visible even for incidental/unsupported responses.
        direct_weights = normalize(shares.where(classes.eq("direct"), 0))
        if d > 0 and not direct_weights.empty:
            for cell, weight in direct_weights.items():
                producer = node(f"direct::{cell}", fmt(cell), 1, "#7F909C", "Input-receptor-high producer", cell)
                contribution = d * weight
                color = production_color(sign, contribution)
                edge(query_id, producer, contribution, row, "input-compatible producer", color, "Cell allocation within the saved direct fraction.")
                edge(producer, product_id, contribution, row, "direct attributed change", color)
        elif d > 0:
            gap = node("direct-missing", "Direct producer<br>unassigned", 1, "#BB913E", "Direct share exists but cell attribution is unavailable.")
            edge(query_id, gap, d, row, "direct attribution unavailable", amber)
            edge(gap, product_id, d, row, "unassigned direct change", amber)

        indirect_weights = normalize(shares.where(classes.eq("indirect"), 0))
        resolved = pd.Series(dtype=float)
        intermediate = row.best_intermediate
        route_note = "No accepted intermediate route."
        # Use the accepted route's saved producer overlap, not all indirect production.
        if i > 0 and pd.notna(intermediate):
            route = row.get("candidate_intermediate_details", {})
            route = route.get(intermediate, {}) if isinstance(route, dict) else {}
            overlap = pd.Series(route.get("overlap_by_producer", {}), dtype=float)
            first_key, second_key = (query_cyto, intermediate), (intermediate, product)
            first_shares, first_classes = shares_and_classes(first_key)
            primary = normalize(first_shares.where(first_classes.eq("direct"), 0))
            saved_overlap = row.get("best_intermediate_producer_overlap", np.nan)
            overlap_weights = normalize(overlap.where(overlap.index.isin(indirect_weights.index), 0))
            if (first_key in lookup and second_key in lookup and not primary.empty and not overlap_weights.empty
                    and pd.notna(saved_overlap) and 0 < saved_overlap <= 1
                    and signs.get(lookup[first_key].direction, 0) * signs.get(lookup[second_key].direction, 0) == sign):
                # Cap allocation at each cell's available indirect share; leave all remainder visible.
                resolved = (overlap_weights * i * saved_overlap).reindex(indirect_weights.index, fill_value=0)
                resolved = resolved.clip(upper=indirect_weights * i)
                flow = float(resolved.sum())
                first_sign, second_sign = signs[lookup[first_key].direction], signs[lookup[second_key].direction]
                middle_id = node(f"intermediate::{product}::{intermediate}", f"{'↑' if first_sign > 0 else '↓'} {fmt(intermediate)}", 2,
                    "#8660A0", "Accepted intermediate for this downstream cytokine.")
                for cell, weight in primary.items():
                    producer = node(f"direct::{cell}", fmt(cell), 1, "#7F909C", "Input-receptor-high producer", cell)
                    contribution = flow * weight
                    color = production_color(first_sign, contribution)
                    edge(query_id, producer, contribution, row, "input-compatible intermediate producer", color, intermediate=intermediate)
                    edge(producer, middle_id, contribution, row, "intermediate attributed change", color, intermediate=intermediate)
                for cell, value in resolved.items():
                    if value <= 0:
                        continue
                    producer = node(f"relay::{product}::{intermediate}::{cell}", f"{fmt(cell)}<br><sup>via {fmt(intermediate)}</sup>", 3, "#8A7A99",
                        "Input-receptor-low / intermediate-receptor-high producer", cell)
                    color = production_color(second_sign, value)
                    edge(middle_id, producer, value, row, "accepted intermediate route", color, intermediate=intermediate)
                    edge(producer, product_id, value, row, "relay attributed change", color, intermediate=intermediate)
                route_note = "Indirect production outside the displayed accepted producer overlap."
            else:
                route_note = "Accepted intermediate is recorded, but complete cell-level route evidence is unavailable."

        remainder = (indirect_weights * i).subtract(resolved, fill_value=0).clip(lower=0)
        unresolved_flow = float(remainder.sum()) if not indirect_weights.empty else i
        if unresolved_flow > 1e-12:
            gap = node(f"unidentified-intermediate::{product}", "Intermediate<br>unidentified", 2, "#BB913E",
                "Input-to-producer connection unresolved for this downstream cytokine.")
            edge(query_id, gap, unresolved_flow, row, "intermediate unresolved", unresolved_color(unresolved_flow), route_note)
            if not indirect_weights.empty:
                for cell, value in remainder.items():
                    if value <= 1e-12:
                        continue
                    producer = node(f"indirect::{product}::{cell}", f"{fmt(cell)}<br><sup>route unresolved</sup>", 3,
                        "#B29A6B", "Input-receptor-low producer; mediator unknown for this downstream cytokine.", cell)
                    edge(gap, producer, value, row, "unresolved input-to-producer route", unresolved_color(value), route_note)
                    edge(producer, product_id, value, row, "indirect attributed change", production_color(sign, value), route_note)
            else:
                gap_producer = node("indirect-missing", "Indirect producer<br>unassigned", 3, "#BB913E", "Indirect share exists but producer cell attribution is unavailable.")
                edge(gap, gap_producer, unresolved_flow, row, "indirect attribution unavailable", amber)
                edge(gap_producer, product_id, unresolved_flow, row, "unassigned indirect change", amber)
        if not attribution_known:
            gap = node("unknown-attribution", "Producer / receptor<br>attribution unknown", 3, "#BB913E", "Saved direct/indirect shares are unavailable.")
            edge(query_id, gap, 1., row, "producer attribution unresolved", amber)
            edge(gap, product_id, 1., row, "observed change; producer unresolved", colors[sign])

        # Compatible recipients are never silently dropped because their input receptor is high.
        support = support_by_pair.get(key)
        if support is None:
            recipients = pd.Series(dtype=float)
            response_note = "Recipient evidence unavailable"
        else:
            support = pd.Series(support, dtype="boolean").fillna(False).astype(bool)
            receptor_classes = pd.Series(producer_details_by_pair.get(key, {}).get("producer_receptor_classes", {}), dtype=object)
            role_by_class = {'direct': 'facultative', 'indirect': 'obligate', 'missing': 'unknown'}
            roles = receptor_classes.reindex(support.index).fillna('missing').map(role_by_class)
            if recipient_role != 'all':
                support = support & roles.eq(recipient_role)
            similarity = pd.Series(similarity_by_pair.get(key, {}), dtype=float).reindex(support.index)
            recipients = normalize((sign * similarity).where(support, 0))
            response_note = (f"No {recipient_role} recipient" if recipient_role != 'all' and not support.any()
                             else "No supported recipient" if not support.any()
                             else "Recipient similarity weights unavailable")
        receptor_classes = pd.Series(producer_details_by_pair.get(key, {}).get("producer_receptor_classes", {}), dtype=object)
        for cell, weight in recipients.items():
            rclass = receptor_classes.get(cell, "missing")
            role = {"direct": ("facultative", "facultative secondary"), "indirect": ("obligate", "obligate secondary")}.get(
                rclass, ("unknown", "input receptor unknown"))
            aligned_similarity = float(sign * similarity.loc[cell])
            recipient = node(f"recipient::{product}::{rclass}::{cell}",
                fmt(cell), 5,
                "#398C83" if rclass == "indirect" else "#8AA6A2" if rclass == "direct" else "#BB913E",
                f"{role[1]}; aligned response similarity r={aligned_similarity:.2f}.", cell,
                recipient_role=role[0])
            saturation = min(max(aligned_similarity / similarity_reference, 0), 1)
            response_color = f"hsl(174, {18 + 68 * saturation:.0f}%, {42 - 10 * saturation:.0f}%)"
            edge(product_id, recipient, weight, row, "supported response similarity", response_color, role[1],
                 raw_similarity=aligned_similarity)
        if recipients.empty:
            sink = node(f"recipient-gap::{response_note}", response_note.replace(" ", "<br>", 1), 5, "#B4B4B4", response_note)
            edge(product_id, sink, 1., row, "recipient support unresolved", neutral, response_note)

    paths = pd.DataFrame(edges)
    if layout == 'publication':
        return _publication_flow_figure(query_cyto,pairs,paths,nodes,mode,figsize_cm,font_size,title,
            minimum_producer_share,minimum_relay_share,minimum_recipient_similarity,stage_x)
    if layout == 'tiny':
        return _compact_flow_figure(query_cyto, pairs, paths, nodes, config, mode, label_formatter, figsize_cm, font_size, title, max_producers, max_recipients)
    node_table = pd.DataFrame(nodes.values())
    node_table["sort_order"] = node_table.cell_type.map(order).fillna(len(order))
    node_table = node_table.sort_values(["stage", "sort_order", "label"]).reset_index(drop=True)
    stage_x = {0: .005, 1: .18, 2: .37, 3: .55, 4: .75, 5: .995}
    node_table["x"] = node_table.stage.map(stage_x)
    node_table["y"] = .5
    for _, ids in node_table.groupby("stage").groups.items():
        node_table.loc[ids, "y"] = [.5] if len(ids) == 1 else np.linspace(.04, .96, len(ids))
    indices = dict(zip(node_table.node_id, node_table.index))
    paths["source_index"], paths["target_index"] = paths.source.map(indices), paths.target.map(indices)
    custom = paths[["secondary", "segment", "causal_class", "confidence", "priority", "detail"]].copy()
    for col in custom:
        custom[col] = custom[col].map(lambda value: escape(str(value)))
    fig = go.Figure(go.Sankey(arrangement="snap", node=dict(label=node_table.label, color=node_table.color,
        x=node_table.x, y=node_table.y, customdata=node_table.detail, pad=13, thickness=12,
        line=dict(color="#444444", width=.4), hovertemplate="%{label}<br>%{customdata}<extra></extra>"),
        link=dict(source=paths.source_index, target=paths.target_index, value=paths.value, color=paths.color, customdata=custom.to_numpy(),
        hovertemplate="%{source.label} → %{target.label}<br>Final candidate: %{customdata[0]}<br>%{customdata[1]}<br>Display share: %{value:.3f}<br>%{customdata[2]}<br>%{customdata[3]}<br>Priority: %{customdata[4]}<br>%{customdata[5]}<extra></extra>")))
    if figsize_cm is None:
        count = int(node_table.groupby("stage").size().max())
        figsize_cm = (30, max(15, count * .85 + 5))
    legend = []
    for causal_class in pairs.causal_class.unique():
        style = config.causal_styles.get(causal_class, {})
        legend.append(f"{_symbol(style, _white_inner_set(config))} {escape(style.get('label', causal_class))}".strip())
    annotations = [dict(x=stage_x[stage], y=1.08, xref="paper", yref="paper", xanchor="left" if stage != 5 else "right",
        showarrow=False, text=f"<b>{label}</b>") for stage, label in enumerate([
        "Input", "Input-responsive<br>producer", "Intermediate / gap", "Relay / unresolved<br>producer", "Downstream cytokine", "Recipient evidence"])]
    annotations += [dict(x=0, y=-.065, xref="paper", yref="paper", xanchor="left", yanchor="top", showarrow=False,
        align="left", text="Red / blue: increased / reduced production. Amber: unresolved route. Teal: supported response similarity.<br>"
             "Each candidate = 1 display unit; D/I medians split production, donor means split producer cells, aligned similarity splits recipients.<br>"
             "Widths summarize evidence, not molecular flux. A shown production route does not establish mediation of the recipient response."),
        dict(x=0, y=-.17, xref="paper", yref="paper", xanchor="left", yanchor="top", showarrow=False,
             align="left", text="<br>".join(" · ".join(legend[i:i+3]) for i in range(0, len(legend), 3)))]
    fig.update_layout(title=dict(text=title or f"{fmt(query_cyto)}: {'all candidates' if mode == 'all' else 'supported response similarity'} ({len(pairs)})", x=.01),
        width=int(figsize_cm[0] / 2.54 * 96), height=int(figsize_cm[1] / 2.54 * 96),
        margin=dict(l=12, r=12, t=90, b=150), font=dict(family="Arial", size=font_size, color="#252525"),
        annotations=annotations, paper_bgcolor="white", meta=dict(mode=mode, candidate_count=len(pairs), width_basis="one unit per candidate", nodes=node_table.to_dict("records")))
    return fig, pairs, paths
