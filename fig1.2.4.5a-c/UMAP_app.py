# panel serve UMAP_app.py --show --autoreload

import numpy as np
import pandas as pd
import panel as pn
import holoviews as hv
import datashader as ds
import anndata as ad
from io import BytesIO
from holoviews.operation.datashader import rasterize
from holoviews.streams import BoundsXY, Tap
from holoviews.selection import link_selections
hv.extension("bokeh")
pn.extension()  # default comms; safe for server mode

# ------- CRITICAL -------
alt_flag = False # use an alternative cell type set
# ------- CRITICAL -------
if not alt_flag:
    from constants import CELL_TYPE_COLORS, ALL_CELL_TYPES
else:
    from constants import CELL_TYPE_COLORS_ALT, ALL_CELL_TYPES_ALT
    ALL_CELL_TYPES = ALL_CELL_TYPES_ALT
    CELL_TYPE_COLORS = CELL_TYPE_COLORS_ALT
    
if not alt_flag:
    relevant_cell_types = [
        "CD4",
        "CD8",
        "B",
        "CD56-bright NK",
        "Treg",
        "MAIT",
        "CD14 Mono",
        "CD16 Mono",
        "CD56-dim NK",
        "NKT",
        "cDC",
        "pDC",
        "other",
        "ILC",
        "HSPC",
        "Plasmablast",
        "Platelet",
    ]
    relevant_cell_type_conversion = {
        "B": "B cell",
        "CD4": "CD4 T cell",
        "CD8": "CD8 T cell",
        "Treg": "Treg",
        "MAIT": "MAIT",
        "NKT": "NKT",
        "cDC": "cDC",
        "pDC": "pDC",
        "CD56-bright NK": "NK CD56hi",
        "CD56-dim NK": "NK CD56low",
        "CD14 Mono": "CD14 Mono",
        "CD16 Mono": "CD16 Mono",
        "HSPC": "HSPC",
        "other": "granulocyte",
        "ILC": "ILC",
        "Plasmablast": "Plasmablast",
        "Platelet": "Platelet",
    }
else:
    relevant_cell_types = [
        "CD4 Naive",
        "CD4 Memory",
        "CD8 Naive",
        "CD8 Memory",
        "B Naive",
        "B Intermediate Memory",
        "Treg",
        "MAIT",
        "CD14 Mono",
        "CD16 Mono",
        "CD56-bright NK",
        "CD56-dim NK",
        "NKT",
        "cDC",
        "pDC",
        "other",
        "ILC",
        "HSPC",
    ]
    relevant_cell_type_conversion = {
        "CD4 Naive": "CD4 Naive",
        "CD4 Memory": "CD4 Memory",
        "CD8 Naive": "CD8 Naive",
        "CD8 Memory": "CD8 Memory",
        "B Naive": "B Naive",
        "B Intermediate Memory": "B Intermediate",
        "Treg": "Treg",
        "MAIT": "MAIT",
        "NKT": "NKT",
        "cDC": "cDC",
        "pDC": "pDC",
        "CD56-bright NK": "NK CD56hi",
        "CD56-dim NK": "NK CD56low",
        "CD14 Mono": "CD14 Mono",
        "CD16 Mono": "CD16 Mono",
        "HSPC": "HSPC",
        "other": "granulocyte",
        "ILC": "ILC",
    }    


umap_file = "../../data/adata_umap_compressed.h5ad"
adata = ad.read_h5ad(umap_file)
adata.obs['donor'] = adata.obs['donor'].str.replace("Donor", "D")

adata.obs.drop(columns=["cell_type"], inplace=True, errors="ignore")
if alt_flag:
    adata.obs['cell_type'] = adata.obs['cell_type_refined']
else:
    adata.obs['cell_type'] = adata.obs['cell_type_refined_coarse']
unique_cell_types = adata.obs['cell_type'].unique()
for cell_type in unique_cell_types:
    if not cell_type in relevant_cell_types:
        raise ValueError(f"Cell type {cell_type} not in relevant_cell_types")
adata.obs['cell_type'] = adata.obs['cell_type'].map(relevant_cell_type_conversion)

# 1) UMAP coords as dense float32, same length as obs
if "X_umap" not in adata.obsm:
    raise KeyError("adata.obsm['X_umap'] is missing.")
coords = np.asarray(adata.obsm["X_umap"], dtype=np.float32)
if coords.ndim != 2 or coords.shape[1] < 2:
    raise ValueError("X_umap must be 2D with at least 2 columns.")
n = coords.shape[0]
if n != adata.n_obs:
    raise ValueError(f"UMAP rows ({n}) != n_obs ({adata.n_obs}).")

# 2) Decide which cell-type column to use (fallback if refined is missing)
CELLTYPE_COL = "cell_type" if "cell_type" in adata.obs.columns else (
               "cell_type" if "cell_type" in adata.obs.columns else None)
if CELLTYPE_COL is None:
    # create an empty string column if nothing is available
    adata.obs["cell_type"] = pd.Series("", index=adata.obs_names, dtype="string")
    CELLTYPE_COL = "cell_type"

COLS = {
    "donor": "donor" if "donor" in adata.obs.columns else None,
    "cell_type": CELLTYPE_COL,
    "stim": "cytokine" if "cytokine" in adata.obs.columns else None,
}

def safe_series(colname: str) -> pd.Series:
    """
    Return a StringDtype Series aligned to adata.obs_names.
    If the desired column is missing, return an empty string series (aligned).
    """
    src = COLS[colname]
    if src is None:
        return pd.Series("", index=adata.obs_names, dtype="string")
    s = adata.obs[src]
    # Ensure Series, aligned index, StringDtype
    if not isinstance(s, pd.Series):
        s = pd.Series(s, index=adata.obs_names)
    # Convert categorical/object to pandas StringDtype (not Python str)
    s = pd.Series(s.astype("string"), index=adata.obs_names, dtype="string")
    return s

# 3) Build DF with a single, consistent index (avoid implicit reindexing)
df = pd.DataFrame(
    {
        "umap1": coords[:, 0],
        "umap2": coords[:, 1],
        "cell_id": adata.obs_names.astype(str),
        "donor": safe_series("donor"),
        "cell_type": safe_series("cell_type"),
        "stim": safe_series("stim"),
    },
    index=adata.obs_names,  # align all Series/arrays to the same index
)

# 4) Coerce numeric and drop any rows with bad coordinates
df["umap1"] = pd.to_numeric(df["umap1"], errors="coerce").astype("float32")
df["umap2"] = pd.to_numeric(df["umap2"], errors="coerce").astype("float32")
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["umap1", "umap2"])  # don’t send NaN coords to Bokeh

# 5) Fill missing labels with readable defaults (empty strings → labels)
df["donor"] = df["donor"].fillna("Unassigned").replace("", "Unassigned")
df["cell_type"] = df["cell_type"].fillna("Unassigned").replace("", "Unassigned")
df["stim"] = df["stim"].fillna("Unstim").replace("", "Unstim")

# 6) Helpful compact dtypes to keep memory small for huge datasets
#    (StringDtype stays; downcast coordinates already float32)
for col in ["donor", "cell_type", "stim"]:
    # If many repeats, categorical can reduce transfer size to browser
    df[col] = df[col].astype("category")

# 7) Widget options helper (works with category or string)
def options_from(col: str):
    s = df[col]
    # .astype("string") will handle category/string; drop empty/na tokens
    vals = s.astype("string").dropna().unique().tolist()
    return sorted([v for v in vals if v and v != "nan"])

# fixed ranges
pad=0.02
x0,x1 = float(np.nanmin(df.umap1)), float(np.nanmax(df.umap1))
y0,y1 = float(np.nanmin(df.umap2)), float(np.nanmax(df.umap2))
xpad,ypad=(x1-x0)*pad,(y1-y0)*pad
XR=(x0-xpad,x1+xpad); YR=(y0-ypad,y1+ypad)

def options_from(col):
    s=df[col].astype("string").dropna()
    return sorted([x for x in s.unique().tolist() if x and x!="nan"])

# widgets
donor_w  = pn.widgets.MultiChoice(name="Donor", options=options_from("donor"))
ctype_w  = pn.widgets.MultiChoice(name="Cell type", options=options_from("cell_type"))
stim_w   = pn.widgets.MultiChoice(name="Stimulation", options=options_from("stim"))
color_w  = pn.widgets.Select(name="Color by", value="cell_type", options=["cell_type","donor","stim","none"])
sample_w = pn.widgets.IntSlider(name="Colored sample per group", start=0, end=2000, value=800, step=100)
size_w   = pn.widgets.IntSlider(name="Point size (sampled)", start=1, end=8, value=2)
alpha_w  = pn.widgets.FloatSlider(name="Sample alpha", start=0.1, end=1.0, value=0.8, step=0.05)
width_w  = pn.widgets.IntInput(name="Plot width", value=900)
height_w = pn.widgets.IntInput(name="Plot height", value=550)
run_btn  = pn.widgets.Button(name="Apply filters", button_type="primary", width=160)

plot_pane  = pn.pane.HoloViews(sizing_mode="stretch_both")
table_widget = pn.widgets.Tabulator(pd.DataFrame(columns=["cell_id","donor","cell_type","stim"]),
                                    height=260, pagination="remote", page_size=200,
                                    selectable=True, show_index=False)

# --- Export controls ---
fn_input   = pn.widgets.TextInput(name="File name", value="selection", placeholder="e.g. my_subset")
fmt_select = pn.widgets.Select(name="Format", options=["csv", "xlsx"], value="csv")

def _build_file():
    df = table_widget.value
    if df is None or len(df) == 0:
        df = pd.DataFrame(columns=["cell_id","donor","cell_type","stim"])

    bio = BytesIO()
    if fmt_select.value == "csv":
        # Write UTF-8 CSV bytes into a BytesIO buffer
        bio.write(df.to_csv(index=False).encode("utf-8"))
    else:  # xlsx
        with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="selection")
    bio.seek(0)
    return bio  # <-- return a file-like object, not raw bytes

file_download = pn.widgets.FileDownload(
    label="Download",
    filename="selection.csv",
    button_type="success",
    callback=_build_file,
)

def _update_filename(event=None):
    base = (fn_input.value or "selection").strip()
    ext  = fmt_select.value
    file_download.filename = f"{base}.{ext}"

fn_input.param.watch(_update_filename, "value")
fmt_select.param.watch(_update_filename, "value")
_update_filename()

# --- selection streams & panes ---
from holoviews.streams import BoundsXY, Tap

summary_pane = pn.pane.Markdown("No selection yet.", height=120)
detail_pane  = pn.pane.Markdown("", height=150)

bounds_stream = BoundsXY(bounds=None)
tap_stream    = Tap(x=None, y=None)

current_filtered = {"df": df}
table_cols = ["cell_id","donor","cell_type","stim"]

def filter_df(donors, ctypes, stims):
    f=df
    if donors: f=f[f["donor"].isin(donors)]
    if ctypes: f=f[f["cell_type"].isin(ctypes)]
    if stims:  f=f[f["stim"].isin(stims)]
    return f

def grouped_sample(f, by, n):
    if n<=0 or by=="none": return f.iloc[0:0]
    if by not in f.columns: return f.sample(min(n,len(f)), random_state=0)
    return (f.groupby(by, group_keys=False)
              .apply(lambda x: x.sample(min(n,len(x)), random_state=0)))

# --- selection helpers & callbacks ---
def summarize(sel: pd.DataFrame) -> str:
    n = len(sel)
    if n == 0:
        return "No cells selected."
    top_ct = sel["cell_type"].value_counts().head(5).to_dict()
    top_dn = sel["donor"].value_counts().head(5).to_dict()
    top_st = sel["stim"].value_counts().head(5).to_dict()
    fmt = lambda d: ", ".join([f"{k} ({v})" for k,v in d.items()]) if d else "—"
    return (
        f"**Selected cells:** {n}\n\n"
        f"**Top cell types:** {fmt(top_ct)}\n\n"
        f"**Top donors:** {fmt(top_dn)}\n\n"
        f"**Top stim:** {fmt(top_st)}"
    )

def apply_bounds_selection(df_sel: pd.DataFrame, bounds):
    if not bounds:
        return df_sel
    x0, y0, x1, y1 = bounds
    lo_x, hi_x = (x0, x1) if x0 <= x1 else (x1, x0)
    lo_y, hi_y = (y0, y1) if y0 <= y1 else (y1, y0)
    mask = df_sel["umap1"].between(lo_x, hi_x) & df_sel["umap2"].between(lo_y, hi_y)
    return df_sel.loc[mask]

def on_bounds_change(**_):
    f = current_filtered["df"]
    sel = apply_bounds_selection(f, bounds_stream.bounds)
    table_widget.value = sel.loc[:, table_cols]
    summary_pane.object = summarize(sel)

def on_tap(x=None, y=None, **_):
    if x is None or y is None:
        return
    f = current_filtered["df"]
    if f.empty:
        detail_pane.object = ""
        return
    dx = f["umap1"].to_numpy() - float(x)
    dy = f["umap2"].to_numpy() - float(y)
    i = int(np.argmin(dx*dx + dy*dy))
    r = f.iloc[i]
    detail_pane.object = (
        f"**Cell:** `{r.cell_id}`\n\n"
        f"- **UMAP:** ({r.umap1:.3f}, {r.umap2:.3f})\n"
        f"- **Donor:** {r.donor}\n"
        f"- **Cell type:** {r.cell_type}\n"
        f"- **Stim:** {r.stim}"
    )

bounds_stream.add_subscriber(on_bounds_change)
tap_stream.add_subscriber(on_tap)

def build_view(event=None):
    w,h=int(width_w.value), int(height_w.value)
    f=filter_df(donor_w.value, ctype_w.value, stim_w.value)

    current_filtered["df"] = f  # expose to callbacks

    LS=link_selections.instance()
    pts = hv.Points(f, kdims=["umap1","umap2"],
                    vdims=["cell_id","donor","cell_type","stim"]).redim.range(umap1=XR, umap2=YR)

    ras = rasterize(pts, aggregator=ds.count(), width=w, height=h,
                    x_range=XR, y_range=YR, precompute=True).opts(
        width=w, height=h, cmap="Greys", colorbar=True,
        tools=["pan","wheel_zoom","box_zoom","reset","tap","lasso_select","box_select"],
        active_tools=["wheel_zoom"], framewise=False
    )

    samp=grouped_sample(f, color_w.value, sample_w.value)
    if color_w.value=="none":
        overlay=hv.Points(samp, kdims=["umap1","umap2"],
                          vdims=["cell_id","donor","cell_type","stim"]).opts(
            size=size_w.value, alpha=alpha_w.value, color="navy", tools=["hover","tap"]
        )
    else:
        overlay=hv.Points(samp, kdims=["umap1","umap2"],
                          vdims=["cell_id","donor","cell_type","stim", color_w.value]).opts(
            color=color_w.value, size=size_w.value, alpha=alpha_w.value, cmap="Category10",
            legend_position="right", tools=["hover","tap"]
        )

    # attach streams
    bounds_stream.source = ras
    tap_stream.source    = ras

    plot_pane.object=(LS(ras*overlay)).opts(shared_axes=True)
    table_widget.value=f[table_cols]
    summary_pane.object = summarize(f)
    detail_pane.object  = ""

run_btn.on_click(build_view)
build_view()

# --- Controls ---
controls = pn.WidgetBox(
    "### Filters", donor_w, ctype_w, stim_w,
    "### Display", color_w, sample_w, size_w, alpha_w, width_w, height_w,
    run_btn, sizing_mode="fixed", width=280  # fixed-width sidebar helps plot stay wide
)

# Optional: keep plot from collapsing
plot_pane.sizing_mode = "stretch_both"
plot_pane.min_width = 700
plot_pane.min_height = 450

# Export bar (filename + format + download button)
export_row = pn.Row(fn_input, fmt_select, file_download, sizing_mode="stretch_width")

right_col = pn.Column(
    plot_pane,
    pn.Tabs(("Selection summary", summary_pane), ("Cell details (tap)", detail_pane)),
    table_widget,
    export_row,                    # <--- NEW
    sizing_mode="stretch_both"
)

inner = pn.Row(controls, right_col, sizing_mode="stretch_both")
dashboard = pn.Column(inner, max_width=1200, sizing_mode="stretch_height")
dashboard.servable()