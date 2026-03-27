from __future__ import annotations

import html
import json
import random
from pathlib import Path
from typing import Any, Dict

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.decomposition import PCA

from .plotting import PALETTE
from .utils import safe_mkdir


def _gene_expression_df(adata) -> pd.DataFrame:
    """Return transcriptome matrix as a sample-by-gene DataFrame."""
    return pd.DataFrame(
        np.asarray(adata.X, dtype=np.float32),
        index=adata.obs_names.astype(str),
        columns=adata.var_names.astype(str),
    )


def _metabolomics_df(adata) -> pd.DataFrame:
    """Return metabolomics matrix as a sample-by-metabolite DataFrame."""
    metab_df = adata.obsm.get("metabolomics_scaled", adata.obsm.get("metabolomics"))
    if isinstance(metab_df, pd.DataFrame):
        return metab_df.copy()
    return pd.DataFrame(
        np.asarray(metab_df, dtype=np.float32),
        index=adata.obs_names.astype(str),
        columns=[str(x) for x in adata.uns.get("metabolite_names", [])],
    )


def _pick_display_features(engine, top_genes: int, top_metabolites: int) -> tuple[list[str], list[str]]:
    """Choose compact feature subsets for interactive editors."""
    gene_df = _gene_expression_df(engine.adata)
    metab_df = _metabolomics_df(engine.adata)

    primary_df = _get_primary_key_gene_df(engine)
    if isinstance(primary_df, pd.DataFrame) and not primary_df.empty:
        gene_candidates = [g for g in primary_df["Gene"].astype(str).tolist() if g in gene_df.columns]
    else:
        gene_candidates = []

    if len(gene_candidates) < top_genes:
        variance_rank = gene_df.var(axis=0).sort_values(ascending=False).index.astype(str).tolist()
        gene_candidates.extend([g for g in variance_rank if g not in gene_candidates])
    selected_genes = gene_candidates[:top_genes]

    summary_df = engine.ml_results.get("metabolite_summary", pd.DataFrame())
    if isinstance(summary_df, pd.DataFrame) and not summary_df.empty and "Metabolite" in summary_df.columns:
        sort_cols = [col for col in ["RRA_Genes", "Candidate_Genes_PCC"] if col in summary_df.columns]
        if sort_cols:
            summary_df = summary_df.sort_values(sort_cols, ascending=[False] * len(sort_cols))
        metabolite_candidates = [m for m in summary_df["Metabolite"].astype(str).tolist() if m in metab_df.columns]
    else:
        metabolite_candidates = []

    if len(metabolite_candidates) < top_metabolites:
        variance_rank = metab_df.var(axis=0).sort_values(ascending=False).index.astype(str).tolist()
        metabolite_candidates.extend([m for m in variance_rank if m not in metabolite_candidates])
    selected_metabs = metabolite_candidates[:top_metabolites]

    return selected_genes, selected_metabs


def _get_primary_key_gene_df(engine) -> pd.DataFrame:
    """Return the key-gene table for the configured primary strategy."""
    strategy = str(getattr(engine.config, "grn_primary_strategy", "rra")).lower()
    return engine.ml_results.get(f"key_genes_{strategy}", pd.DataFrame())


def _module_color_map(modules: list[str]) -> Dict[str, str]:
    """Create a stable module-to-color mapping."""
    unique_modules = [module for module in sorted(set(modules)) if module != "Unassigned"]
    palette = cm.get_cmap("tab20")(np.linspace(0, 1, num=max(1, len(unique_modules))))
    color_map = {module: mcolors.to_hex(palette[idx]) for idx, module in enumerate(unique_modules)}
    color_map["Unassigned"] = "#bdbdbd"
    return color_map


def _json_default(obj: Any) -> Any:
    """Convert NumPy / pandas scalars into JSON-friendly Python values."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Index, pd.Series)):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")


def _json_dumps(data: Any) -> str:
    """Serialize payloads with UTF-8-friendly JSON output."""
    return json.dumps(data, ensure_ascii=False, default=_json_default)


def _build_summary_payload(engine, cfg) -> dict[str, Any]:
    """Build a compact payload for the report header."""
    grn_edges_df = engine.ml_results.get("grn_edges_df", pd.DataFrame())
    module_summary = engine.wgcna_results.get("Module_Summary", pd.DataFrame())
    return {
        "projectName": str(cfg.project_name),
        "samples": int(engine.adata.n_obs),
        "genes": int(engine.adata.n_vars),
        "metabolites": int(len(engine.adata.uns.get("metabolite_names", []))),
        "grnEdges": int(len(grn_edges_df)) if isinstance(grn_edges_df, pd.DataFrame) else 0,
        "wgcnaModules": int(len(module_summary)) if isinstance(module_summary, pd.DataFrame) else 0,
        "selectedPower": engine.wgcna_results.get("Selected_Power", "NA"),
    }


def _build_pca_payload(matrix, sample_names, title: str, cfg) -> dict[str, Any] | None:
    """Prepare a lightweight PCA scatter payload for the browser."""
    if isinstance(matrix, pd.DataFrame):
        values = matrix.to_numpy(dtype=float, copy=False)
    else:
        values = np.asarray(matrix, dtype=float)

    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        return None

    sample_names = [str(name) for name in sample_names]
    if len(sample_names) != values.shape[0]:
        return None

    X = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    pca = PCA(n_components=2, random_state=cfg.random_state)
    coords = pca.fit_transform(X)
    var_exp = pca.explained_variance_ratio_ * 100.0

    return {
        "title": title,
        "width": 940,
        "height": 680,
        "xLabel": f"PC1 ({var_exp[0]:.1f}%)",
        "yLabel": f"PC2 ({var_exp[1]:.1f}%)",
        "points": [
            {
                "name": name,
                "x": float(x),
                "y": float(y),
            }
            for name, (x, y) in zip(sample_names, coords)
        ],
    }


def _build_correlation_circle_payload(engine, cfg) -> dict[str, Any] | None:
    """Prepare correlation-circle data for the browser editor."""
    gene_names, metabolite_names = _pick_display_features(
        engine,
        top_genes=cfg.correlation_circle_top_genes,
        top_metabolites=cfg.correlation_circle_top_metabolites,
    )
    if len(gene_names) < 2 or len(metabolite_names) < 1:
        return None

    gene_df = _gene_expression_df(engine.adata).loc[:, gene_names]
    metab_df = _metabolomics_df(engine.adata).loc[:, metabolite_names]
    combined = pd.concat([gene_df, metab_df], axis=1)
    if combined.shape[0] < 3 or combined.shape[1] < 3:
        return None

    X = combined.to_numpy(dtype=float, copy=False)
    Xz = np.nan_to_num(zscore(X, axis=0, ddof=1), nan=0.0, posinf=0.0, neginf=0.0)
    pca = PCA(n_components=2, random_state=cfg.random_state)
    scores = pca.fit_transform(Xz)
    score_z = np.nan_to_num(zscore(scores, axis=0, ddof=1), nan=0.0, posinf=0.0, neginf=0.0)
    corr_coords = (Xz.T @ score_z) / max(1, Xz.shape[0] - 1)
    var_exp = pca.explained_variance_ratio_ * 100.0

    palette = {"Gene": PALETTE["gene"], "Metabolite": PALETTE["metabolite"]}
    items = []
    feature_types = ["Gene"] * len(gene_names) + ["Metabolite"] * len(metabolite_names)
    for idx, (name, feature_type) in enumerate(zip(combined.columns.astype(str), feature_types), start=1):
        x = float(np.clip(corr_coords[idx - 1, 0], -1.05, 1.05))
        y = float(np.clip(corr_coords[idx - 1, 1], -1.05, 1.05))
        items.append(
            {
                "id": f"{feature_type.lower()}_{idx:03d}",
                "label": str(name),
                "type": feature_type,
                "x": x,
                "y": y,
                "labelDx": 14 if x >= 0 else -14,
                "labelDy": -10 if y >= 0 else 14,
                "color": palette[feature_type],
            }
        )

    return {
        "title": "Correlation Circle Editor",
        "subtitle": "Drag arrow endpoints to adjust vector placement, drag labels separately to declutter text, and export the polished result.",
        "width": 940,
        "height": 780,
        "xLabel": f"PC1 ({var_exp[0]:.1f}%)",
        "yLabel": f"PC2 ({var_exp[1]:.1f}%)",
        "items": items,
    }


def _build_network_payload(engine, cfg) -> dict[str, Any] | None:
    """Prepare a force-layout GRN payload for the browser."""
    edge_df = engine.ml_results.get("grn_edges_df")
    if not isinstance(edge_df, pd.DataFrame) or edge_df.empty:
        return None

    ranked = edge_df.assign(AbsPCC=edge_df["PCC_R"].abs()).sort_values(
        ["Support_Count", "AbsPCC"],
        ascending=[False, False],
    )
    top_edges = ranked.head(cfg.circos_top_edges).copy()
    if top_edges.empty:
        return None

    genes = top_edges["Gene"].astype(str).drop_duplicates().tolist()
    metabolites = top_edges["Metabolite"].astype(str).drop_duplicates().tolist()
    if not genes or not metabolites:
        return None

    module_df = engine.wgcna_results.get("Gene_Modules", pd.DataFrame())
    if isinstance(module_df, pd.DataFrame) and not module_df.empty and {"Gene", "Module"}.issubset(module_df.columns):
        gene_to_module = dict(zip(module_df["Gene"].astype(str), module_df["Module"].astype(str)))
    else:
        gene_to_module = {}

    used_modules = [gene_to_module.get(gene, "Unassigned") for gene in genes]
    module_colors = _module_color_map(used_modules)
    rng = random.Random(int(cfg.random_state))

    width = 1100
    height = 750
    y_min = 70.0
    y_max = float(height - 70)

    nodes = []
    for gene in genes:
        module = gene_to_module.get(gene, "Unassigned")
        nodes.append(
            {
                "id": f"gene::{gene}",
                "label": gene,
                "type": "Gene",
                "module": module,
                "color": module_colors.get(module, "#bdbdbd"),
                "x": float(rng.uniform(100.0, 500.0)),
                "y": float(rng.uniform(y_min, y_max)),
            }
        )

    for metabolite in metabolites:
        nodes.append(
            {
                "id": f"metab::{metabolite}",
                "label": metabolite,
                "type": "Metabolite",
                "module": "Metabolite",
                "color": "#111827",
                "x": float(rng.uniform(600.0, 1000.0)),
                "y": float(rng.uniform(y_min, y_max)),
            }
        )

    node_ids = {node["id"] for node in nodes}
    edges = []
    for edge_idx, (_, row) in enumerate(top_edges.reset_index(drop=True).iterrows(), start=1):
        source = f"gene::{str(row['Gene'])}"
        target = f"metab::{str(row['Metabolite'])}"
        if source not in node_ids or target not in node_ids:
            continue

        corr = float(row["PCC_R"]) if pd.notna(row["PCC_R"]) else 0.0
        support = int(row["Support_Count"]) if pd.notna(row["Support_Count"]) else 1
        abs_corr = min(1.0, abs(corr))
        edges.append(
            {
                "id": f"edge_{edge_idx:03d}",
                "source": source,
                "target": target,
                "correlation": corr,
                "support": support,
                "color": PALETTE["edge_positive"] if corr >= 0 else PALETTE["edge_negative"],
                "width": 1.0 + 2.0 * abs_corr,
                "opacity": min(0.95, 0.25 + 0.18 * support),
            }
        )

    if not edges:
        return None

    module_legend = [
        {"label": module, "color": module_colors[module]}
        for module in sorted(set(used_modules))
        if module in module_colors
    ]

    return {
        "nodes": nodes,
        "edges": edges,
        "moduleLegend": module_legend,
        "width": width,
        "height": height,
    }


def _interactive_html_template() -> str:
    """Return a standalone interactive report template."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>DeepOmics Interactive Figure Studio - __PROJECT_NAME__</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root {
      --bg: #f8fafc;
      --card: #ffffff;
      --border: #dbe4f0;
      --text: #0f172a;
      --muted: #475569;
      --accent: #2563eb;
      --danger: #dc2626;
      --success: #059669;
      --shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
      --tab-shadow: 0 6px 16px rgba(15, 23, 42, 0.06);
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      font-family: "Inter", "Segoe UI", Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }
    .tab-nav {
      position: sticky;
      top: 0;
      z-index: 40;
      background: rgba(255, 255, 255, 0.96);
      backdrop-filter: blur(10px);
      border-bottom: 1px solid var(--border);
      box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
    }
    .tab-nav-inner {
      max-width: 1440px;
      margin: 0 auto;
      padding: 0 28px;
      display: flex;
      gap: 6px;
      overflow-x: auto;
    }
    .tab-btn {
      appearance: none;
      border: none;
      background: transparent;
      color: var(--muted);
      padding: 18px 14px 14px 14px;
      font-size: 14px;
      font-weight: 700;
      cursor: pointer;
      border-bottom: 2.5px solid transparent;
      border-radius: 0;
      box-shadow: none;
      transform: none;
      white-space: nowrap;
    }
    .tab-btn:hover {
      color: var(--text);
      border-color: #bfdbfe;
      box-shadow: none;
      transform: none;
    }
    .tab-btn.active {
      color: var(--accent);
      border-color: var(--accent);
    }
    .page {
      max-width: 1440px;
      margin: 0 auto;
      padding: 28px;
    }
    .tab-panel {
      display: none;
      animation: fadeIn 0.16s ease-out;
    }
    .tab-panel.active {
      display: block;
    }
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(3px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .hero {
      background: linear-gradient(135deg, #eff6ff 0%, #ffffff 100%);
      border: 1px solid var(--border);
      border-radius: 20px;
      padding: 24px 28px;
      box-shadow: var(--shadow);
      margin-bottom: 24px;
    }
    .hero h1 {
      margin: 0 0 8px 0;
      font-size: 30px;
      color: #111827;
    }
    .hero p {
      margin: 0;
      color: var(--muted);
      max-width: 980px;
    }
    .chip-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 16px;
    }
    .chip {
      background: #eef2ff;
      border: 1px solid #c7d2fe;
      border-radius: 999px;
      padding: 6px 12px;
      color: #3730a3;
      font-size: 13px;
      font-weight: 600;
    }
    .callout {
      margin-top: 16px;
      background: #f8fafc;
      border-left: 4px solid var(--accent);
      border-radius: 12px;
      padding: 14px 16px;
      color: var(--muted);
    }
    .guide {
      margin-top: 14px;
      color: #1e3a8a;
      font-weight: 600;
    }
    .grid {
      display: grid;
      gap: 24px;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 20px;
      box-shadow: var(--shadow);
      padding: 20px 20px 16px 20px;
    }
    .card h2 {
      margin: 0 0 6px 0;
      font-size: 24px;
    }
    .card p.desc {
      margin: 0 0 14px 0;
      color: var(--muted);
      font-size: 14px;
    }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 10px;
      margin-bottom: 14px;
    }
    .toolbar-group {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding-right: 10px;
      margin-right: 2px;
      border-right: 1px solid var(--border);
    }
    .toolbar-group:last-child {
      border-right: none;
      padding-right: 0;
      margin-right: 0;
    }
    button {
      appearance: none;
      border: 1px solid var(--border);
      border-radius: 10px;
      background: #ffffff;
      color: var(--text);
      padding: 10px 14px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      transition: 0.15s ease;
    }
    button:hover {
      border-color: #93c5fd;
      transform: translateY(-1px);
      box-shadow: 0 8px 18px rgba(37, 99, 235, 0.10);
    }
    button.primary {
      background: #eff6ff;
      border-color: #93c5fd;
      color: #1d4ed8;
    }
    button.warn {
      background: #fef2f2;
      border-color: #fecaca;
      color: #b91c1c;
    }
    input[type="color"] {
      width: 42px;
      height: 36px;
      padding: 0;
      border: 1px solid var(--border);
      border-radius: 10px;
      background: #ffffff;
      cursor: pointer;
    }
    .canvas-shell {
      background: #ffffff;
      border: 1px solid var(--border);
      border-radius: 18px;
      overflow: hidden;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.65);
    }
    svg {
      width: 100%;
      display: block;
      background: #ffffff;
      touch-action: none;
    }
    svg text {
      user-select: none;
      -webkit-user-select: none;
    }
    .status {
      margin-top: 12px;
      font-size: 13px;
      color: var(--muted);
      min-height: 20px;
    }
    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 14px;
      color: var(--muted);
      font-size: 13px;
    }
    .legend-item {
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }
    .legend-swatch {
      width: 12px;
      height: 12px;
      border-radius: 999px;
      border: 1px solid rgba(15, 23, 42, 0.15);
      flex: none;
    }
    .fallback {
      border: 1px dashed var(--border);
      border-radius: 14px;
      padding: 18px;
      background: #f8fafc;
      color: var(--muted);
    }
    .floating-tooltip {
      position: fixed;
      left: 0;
      top: 0;
      z-index: 120;
      max-width: 320px;
      pointer-events: none;
      opacity: 0;
      transform: translate(0, 0);
      transition: opacity 0.08s ease;
      background: rgba(15, 23, 42, 0.96);
      color: #f8fafc;
      border-radius: 12px;
      padding: 10px 12px;
      font-size: 12px;
      line-height: 1.45;
      box-shadow: 0 14px 32px rgba(15, 23, 42, 0.22);
    }
    .floating-tooltip.visible {
      opacity: 1;
    }
    .empty-state {
      color: var(--muted);
      padding: 14px 4px 0 4px;
      font-size: 14px;
    }
    @media (max-width: 880px) {
      .page { padding: 18px; }
      .tab-nav-inner { padding: 0 18px; }
      .hero { padding: 20px; }
      .card { padding: 16px; }
      .toolbar-group {
        border-right: none;
        padding-right: 0;
        margin-right: 0;
      }
    }
  </style>
</head>
<body>
  <nav class="tab-nav" aria-label="Interactive analysis navigation">
    <div class="tab-nav-inner">
      <button class="tab-btn active" data-tab="overviewPanel" type="button">Overview</button>
      <button class="tab-btn" data-tab="txPcaPanel" type="button">Transcriptome PCA</button>
      <button class="tab-btn" data-tab="metabPcaPanel" type="button">Metabolome PCA</button>
      <button class="tab-btn" data-tab="networkPanel" type="button">GRN Network</button>
      <button class="tab-btn" data-tab="circlePanel" type="button">Correlation Circle</button>
    </div>
  </nav>

  <div class="page">
    <section class="tab-panel active" id="overviewPanel">
      <section class="hero">
        <h1>DeepOmics Interactive Figure Studio</h1>
        <p>This standalone HTML report keeps the underlying DeepOmics model outputs untouched while exposing an editable figure layer for publication polishing. It is intentionally focused on the most annotation-heavy figures: multi-omics ordination panels, the force-directed gene-metabolite network, and the correlation circle editor.</p>
        <div class="chip-row" id="summaryChips"></div>
        <div class="callout">
          Recommended workflow: keep the package-generated CSV / H5AD / static SVG outputs as the reproducible record, then use this page only for final presentation edits such as label deconfliction, node movement, edge pruning, recoloring, and figure export.
        </div>
        <p class="guide">Click the tabs above to explore interactive analysis panels.</p>
      </section>
    </section>

    <section class="tab-panel" id="txPcaPanel">
      <section class="card">
        <h2>Transcriptome PCA</h2>
        <p class="desc">Inspect transcriptome sample separation in two dimensions, recolor the scatter interactively, and export the figure as SVG or PNG.</p>
        <div class="toolbar">
          <div class="toolbar-group">
            <input type="color" id="txPcaColor" value="#4c78a8" aria-label="Transcriptome PCA point color" />
            <button class="primary" id="txPcaApplyColor" type="button">Apply Color</button>
          </div>
          <div class="toolbar-group">
            <button id="txPcaSvgBtn" type="button">Save SVG</button>
            <button id="txPcaPngBtn" type="button">Save PNG</button>
          </div>
        </div>
        <div class="canvas-shell">
          <svg id="txPcaSvg" viewBox="0 0 940 680" role="img" aria-label="Transcriptome PCA scatter plot"></svg>
        </div>
        <div class="status" id="txPcaStatus"></div>
        <div class="fallback" id="txPcaFallback" hidden>No transcriptome PCA payload was generated. This usually means the matrix did not contain enough samples or features for a 2D PCA projection.</div>
      </section>
    </section>

    <section class="tab-panel" id="metabPcaPanel">
      <section class="card">
        <h2>Metabolome PCA</h2>
        <p class="desc">Inspect metabolome sample separation in two dimensions, recolor the scatter interactively, and export the figure as SVG or PNG.</p>
        <div class="toolbar">
          <div class="toolbar-group">
            <input type="color" id="metabPcaColor" value="#4c78a8" aria-label="Metabolome PCA point color" />
            <button class="primary" id="metabPcaApplyColor" type="button">Apply Color</button>
          </div>
          <div class="toolbar-group">
            <button id="metabPcaSvgBtn" type="button">Save SVG</button>
            <button id="metabPcaPngBtn" type="button">Save PNG</button>
          </div>
        </div>
        <div class="canvas-shell">
          <svg id="metabPcaSvg" viewBox="0 0 940 680" role="img" aria-label="Metabolome PCA scatter plot"></svg>
        </div>
        <div class="status" id="metabPcaStatus"></div>
        <div class="fallback" id="metabPcaFallback" hidden>No metabolome PCA payload was generated. This usually means the matrix did not contain enough samples or features for a 2D PCA projection.</div>
      </section>
    </section>

    <section class="tab-panel" id="networkPanel">
      <section class="card">
        <h2>GRN Network</h2>
        <p class="desc">Explore the top prioritized gene-metabolite associations as a force-directed network. Drag a node to pin it, click a node or edge to select it, remove distracting elements, and export the current layout.</p>
        <div class="toolbar">
          <div class="toolbar-group">
            <button class="primary" id="networkRelayoutBtn" type="button">Re-layout</button>
            <button class="warn" id="networkDeleteBtn" type="button">Delete Selected</button>
          </div>
          <div class="toolbar-group">
            <button id="networkSvgBtn" type="button">Save SVG</button>
            <button id="networkPngBtn" type="button">Save PNG</button>
          </div>
        </div>
        <div class="canvas-shell">
          <svg id="networkSvg" viewBox="0 0 1100 750" role="img" aria-label="GRN force-directed network"></svg>
        </div>
        <div class="status" id="networkStatus"></div>
        <div class="fallback" id="networkFallback" hidden>No GRN network payload was generated. This usually means the current run did not produce enough prioritized gene-metabolite edges.</div>
      </section>
    </section>

    <section class="tab-panel" id="circlePanel">
      <section class="card" id="circleCard">
        <h2>Correlation circle editor</h2>
        <p class="desc">Best suited for fixing crowded labels in PCA correlation circles. Drag feature endpoints to move vectors, drag labels independently, double-click labels to rename, add free-text notes, and export the edited figure.</p>
        <div class="toolbar">
          <button class="primary" id="circleRenameBtn" type="button">Rename selected feature</button>
          <button id="circleAddNoteBtn" type="button">Add note</button>
          <button class="warn" id="circleDeleteBtn" type="button">Delete selected item</button>
          <button id="circleResetBtn" type="button">Reset layout</button>
          <button id="circleSvgBtn" type="button">Save SVG</button>
          <button id="circlePngBtn" type="button">Save PNG</button>
        </div>
        <div class="canvas-shell">
          <svg id="circleSvg" viewBox="0 0 940 780" role="img" aria-label="Correlation circle editor"></svg>
        </div>
        <div class="legend">
          <span class="legend-item"><span class="legend-swatch" style="background:#2563eb"></span>Genes</span>
          <span class="legend-item"><span class="legend-swatch" style="background:#dc2626"></span>Metabolites</span>
          <span class="legend-item"><span class="legend-swatch" style="background:#f8fafc;border-color:#94a3b8"></span>Drag labels independently for decluttering</span>
        </div>
        <div class="status" id="circleStatus"></div>
        <div class="fallback" id="circleFallback" hidden>No correlation-circle payload was generated. This usually means the run did not produce enough prioritized features for an informative editor.</div>
      </section>
    </section>
  </div>

  <div class="floating-tooltip" id="floatingTooltip"></div>

  <script>
    const summaryPayload = __SUMMARY_PAYLOAD__;
    const txPcaPayload = __TRANSCRIPTOME_PCA_PAYLOAD__;
    const metabPcaPayload = __METABOLOME_PCA_PAYLOAD__;
    const networkPayload = __NETWORK_PAYLOAD__;
    const circlePayload = __CIRCLE_PAYLOAD__;

    function deepCopy(value) {
      return JSON.parse(JSON.stringify(value));
    }

    function svgEl(tag, attrs = {}) {
      const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
      Object.entries(attrs).forEach(([key, value]) => {
        if (value !== undefined && value !== null) {
          el.setAttribute(key, String(value));
        }
      });
      return el;
    }

    function clearSvg(svg) {
      while (svg.firstChild) {
        svg.removeChild(svg.firstChild);
      }
    }

    function clamp(value, minValue, maxValue) {
      return Math.max(minValue, Math.min(maxValue, value));
    }

    function clientToSvg(svg, clientX, clientY) {
      const point = svg.createSVGPoint();
      point.x = clientX;
      point.y = clientY;
      const ctm = svg.getScreenCTM();
      return ctm ? point.matrixTransform(ctm.inverse()) : { x: clientX, y: clientY };
    }

    function attachSvgDrag(target, svg, handlers = {}) {
      target.style.cursor = "grab";
      target.addEventListener("pointerdown", (event) => {
        event.preventDefault();
        event.stopPropagation();
        let previous = clientToSvg(svg, event.clientX, event.clientY);
        try {
          target.setPointerCapture(event.pointerId);
        } catch (err) {}
        if (handlers.start) {
          handlers.start(event);
        }

        const onMove = (moveEvent) => {
          if (moveEvent.pointerId !== event.pointerId) {
            return;
          }
          const current = clientToSvg(svg, moveEvent.clientX, moveEvent.clientY);
          const delta = {
            x: current.x,
            y: current.y,
            dx: current.x - previous.x,
            dy: current.y - previous.y,
          };
          previous = current;
          if (handlers.move) {
            handlers.move(moveEvent, delta);
          }
        };

        const onEnd = (endEvent) => {
          if (endEvent.pointerId !== event.pointerId) {
            return;
          }
          target.removeEventListener("pointermove", onMove);
          target.removeEventListener("pointerup", onEnd);
          target.removeEventListener("pointercancel", onEnd);
          try {
            target.releasePointerCapture(event.pointerId);
          } catch (err) {}
          if (handlers.end) {
            handlers.end(endEvent);
          }
        };

        target.addEventListener("pointermove", onMove);
        target.addEventListener("pointerup", onEnd);
        target.addEventListener("pointercancel", onEnd);
      });
    }

    function downloadBlob(filename, blob) {
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = filename;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    }

    function serializeSvg(svg) {
      const clone = svg.cloneNode(true);
      clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
      clone.setAttribute("xmlns:xlink", "http://www.w3.org/1999/xlink");
      const serializer = new XMLSerializer();
      return serializer.serializeToString(clone);
    }

    function exportSvg(svg, filename) {
      const payload = serializeSvg(svg);
      downloadBlob(filename, new Blob([payload], { type: "image/svg+xml;charset=utf-8" }));
    }

    function exportPng(svg, filename, scale = 2.5) {
      const svgString = serializeSvg(svg);
      const blob = new Blob([svgString], { type: "image/svg+xml;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const img = new Image();
      img.onload = () => {
        const viewBox = svg.viewBox.baseVal;
        const width = viewBox && viewBox.width ? viewBox.width : svg.clientWidth;
        const height = viewBox && viewBox.height ? viewBox.height : svg.clientHeight;
        const canvas = document.createElement("canvas");
        canvas.width = Math.round(width * scale);
        canvas.height = Math.round(height * scale);
        const ctx = canvas.getContext("2d");
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        canvas.toBlob((pngBlob) => {
          if (pngBlob) {
            downloadBlob(filename, pngBlob);
          }
          URL.revokeObjectURL(url);
        }, "image/png");
      };
      img.onerror = () => {
        URL.revokeObjectURL(url);
      };
      img.src = url;
    }

    const tooltipEl = document.getElementById("floatingTooltip");

    function showTooltip(htmlText, clientX, clientY) {
      tooltipEl.innerHTML = htmlText;
      tooltipEl.style.left = `${clientX + 14}px`;
      tooltipEl.style.top = `${clientY + 14}px`;
      tooltipEl.classList.add("visible");
    }

    function moveTooltip(clientX, clientY) {
      tooltipEl.style.left = `${clientX + 14}px`;
      tooltipEl.style.top = `${clientY + 14}px`;
    }

    function hideTooltip() {
      tooltipEl.classList.remove("visible");
    }

    function fillSummary() {
      const chips = document.getElementById("summaryChips");
      const rows = [
        ["Samples", summaryPayload.samples],
        ["Genes", summaryPayload.genes],
        ["Metabolites", summaryPayload.metabolites],
        ["GRN edges", summaryPayload.grnEdges],
        ["WGCNA modules", summaryPayload.wgcnaModules],
        ["Selected power", summaryPayload.selectedPower],
      ];
      rows.forEach(([label, value]) => {
        const span = document.createElement("span");
        span.className = "chip";
        span.textContent = `${label}: ${value}`;
        chips.appendChild(span);
      });
    }

    function setupTabs() {
      const buttons = Array.from(document.querySelectorAll(".tab-btn"));
      const panels = Array.from(document.querySelectorAll(".tab-panel"));

      function activate(tabId) {
        buttons.forEach((button) => {
          button.classList.toggle("active", button.dataset.tab === tabId);
        });
        panels.forEach((panel) => {
          panel.classList.toggle("active", panel.id === tabId);
        });
        hideTooltip();
      }

      buttons.forEach((button) => {
        button.addEventListener("click", () => activate(button.dataset.tab));
      });
    }

    function initPcaScatter(payload, ids) {
      const svg = document.getElementById(ids.svgId);
      const fallback = document.getElementById(ids.fallbackId);
      const statusEl = document.getElementById(ids.statusId);
      const colorInput = document.getElementById(ids.colorId);
      const applyBtn = document.getElementById(ids.applyBtnId);
      const svgBtn = document.getElementById(ids.svgBtnId);
      const pngBtn = document.getElementById(ids.pngBtnId);

      const setStatus = (message) => {
        statusEl.textContent = message || "";
      };

      if (!payload || !Array.isArray(payload.points) || payload.points.length === 0) {
        fallback.hidden = false;
        svg.hidden = true;
        return;
      }

      const width = Number(payload.width || 940);
      const height = Number(payload.height || 680);
      const margin = { top: 42, right: 34, bottom: 78, left: 88 };
      const plotWidth = width - margin.left - margin.right;
      const plotHeight = height - margin.top - margin.bottom;
      const points = deepCopy(payload.points);
      let pointColor = colorInput ? colorInput.value : "#4c78a8";
      svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

      const xValues = points.map((point) => Number(point.x) || 0);
      const yValues = points.map((point) => Number(point.y) || 0);

      const xRawMin = Math.min(0, ...xValues);
      const xRawMax = Math.max(0, ...xValues);
      const yRawMin = Math.min(0, ...yValues);
      const yRawMax = Math.max(0, ...yValues);

      const xSpan = Math.max(1e-6, xRawMax - xRawMin);
      const ySpan = Math.max(1e-6, yRawMax - yRawMin);

      const xMin = xRawMin - xSpan * 0.12;
      const xMax = xRawMax + xSpan * 0.12;
      const yMin = yRawMin - ySpan * 0.12;
      const yMax = yRawMax + ySpan * 0.12;

      const scaleX = (value) => margin.left + ((value - xMin) / Math.max(1e-6, xMax - xMin)) * plotWidth;
      const scaleY = (value) => margin.top + ((yMax - value) / Math.max(1e-6, yMax - yMin)) * plotHeight;

      function render() {
        clearSvg(svg);
        svg.appendChild(svgEl("rect", { x: 0, y: 0, width, height, fill: "#ffffff" }));

        const chartGroup = svgEl("g");
        svg.appendChild(chartGroup);

        chartGroup.appendChild(svgEl("rect", {
          x: margin.left,
          y: margin.top,
          width: plotWidth,
          height: plotHeight,
          rx: 18,
          fill: "#ffffff",
          stroke: "#e2e8f0",
        }));

        const xZero = scaleX(0);
        const yZero = scaleY(0);

        chartGroup.appendChild(svgEl("line", {
          x1: margin.left,
          y1: yZero,
          x2: margin.left + plotWidth,
          y2: yZero,
          stroke: "#cbd5e1",
          "stroke-width": 1.2,
          "stroke-dasharray": "6 5",
        }));
        chartGroup.appendChild(svgEl("line", {
          x1: xZero,
          y1: margin.top,
          x2: xZero,
          y2: margin.top + plotHeight,
          stroke: "#cbd5e1",
          "stroke-width": 1.2,
          "stroke-dasharray": "6 5",
        }));

        chartGroup.appendChild(svgEl("text", {
          x: width / 2,
          y: height - 24,
          "text-anchor": "middle",
          fill: "#334155",
          "font-size": 14,
          "font-weight": 600,
        })).textContent = payload.xLabel || "PC1";

        const yAxisLabel = svgEl("text", {
          x: 24,
          y: height / 2,
          fill: "#334155",
          "font-size": 14,
          "font-weight": 600,
          transform: `rotate(-90 24 ${height / 2})`,
          "text-anchor": "middle",
        });
        yAxisLabel.textContent = payload.yLabel || "PC2";
        chartGroup.appendChild(yAxisLabel);

        const helper = svgEl("text", {
          x: margin.left,
          y: 24,
          fill: "#64748b",
          "font-size": 12.5,
        });
        helper.textContent = "Hover points to inspect sample names and exact PC coordinates.";
        chartGroup.appendChild(helper);

        const pointLayer = svgEl("g");
        chartGroup.appendChild(pointLayer);

        points.forEach((point) => {
          const cx = scaleX(Number(point.x) || 0);
          const cy = scaleY(Number(point.y) || 0);
          const circle = svgEl("circle", {
            cx,
            cy,
            r: 5,
            fill: pointColor,
            stroke: "#ffffff",
            "stroke-width": 1.4,
            opacity: 0.95,
          });

          circle.addEventListener("mouseenter", (event) => {
            circle.setAttribute("r", "6.2");
            showTooltip(
              `<strong>${String(point.name)}</strong><br/>PC1: ${(Number(point.x) || 0).toFixed(3)}<br/>PC2: ${(Number(point.y) || 0).toFixed(3)}`,
              event.clientX,
              event.clientY
            );
          });
          circle.addEventListener("mousemove", (event) => {
            moveTooltip(event.clientX, event.clientY);
          });
          circle.addEventListener("mouseleave", () => {
            circle.setAttribute("r", "5");
            hideTooltip();
          });

          pointLayer.appendChild(circle);
          point._circleRef = circle;
        });
      }

      applyBtn.addEventListener("click", () => {
        pointColor = colorInput.value || "#4c78a8";
        points.forEach((point) => {
          if (point._circleRef) {
            point._circleRef.setAttribute("fill", pointColor);
          }
        });
        setStatus(`Applied point color ${pointColor} to ${points.length} samples.`);
      });

      svgBtn.addEventListener("click", () => {
        exportSvg(svg, ids.svgFilename);
        setStatus("Saved current PCA view as SVG.");
      });

      pngBtn.addEventListener("click", () => {
        exportPng(svg, ids.pngFilename);
        setStatus("Saved current PCA view as PNG.");
      });

      render();
      setStatus("Ready. Hover any point to inspect the sample coordinates.");
    }

    function initNetwork(payload) {
      const svg = document.getElementById("networkSvg");
      const fallback = document.getElementById("networkFallback");
      const statusEl = document.getElementById("networkStatus");
      const relayoutBtn = document.getElementById("networkRelayoutBtn");
      const deleteBtn = document.getElementById("networkDeleteBtn");
      const svgBtn = document.getElementById("networkSvgBtn");
      const pngBtn = document.getElementById("networkPngBtn");

      const setStatus = (message) => {
        statusEl.textContent = message || "";
      };

      if (!payload || !Array.isArray(payload.nodes) || payload.nodes.length === 0 || !Array.isArray(payload.edges) || payload.edges.length === 0) {
        fallback.hidden = false;
        svg.hidden = true;
        return;
      }

      const width = Number(payload.width || 1100);
      const height = Number(payload.height || 750);
      svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

      const state = {
        nodes: deepCopy(payload.nodes).map((node) => ({
          ...node,
          vx: 0,
          vy: 0,
          pinned: false,
          dragging: false,
          deleted: false,
          visibleLabel: false,
          initialX: Number(node.x),
          initialY: Number(node.y),
        })),
        edges: deepCopy(payload.edges).map((edge) => ({
          ...edge,
          deleted: false,
        })),
        selectedNodeId: null,
        selectedEdgeId: null,
        rafId: null,
        running: false,
        iteration: 0,
        maxIterations: 300,
        repulsionStrength: 5000,
        springStrength: 0.005,
        restLength: 120,
        centerGravity: 0.01,
        damping: 0.85,
        moveThreshold: 0.35,
        showLabelsByDefault: payload.nodes.length <= 50,
        svg,
        refs: {
          edgeLayer: null,
          nodeLayer: null,
          legendLayer: null,
          titleLayer: null,
        },
      };

      function activeNodes() {
        return state.nodes.filter((node) => !node.deleted);
      }

      function activeEdges() {
        return state.edges.filter((edge) => !edge.deleted);
      }

      function getNode(id) {
        return state.nodes.find((node) => node.id === id);
      }

      function getEdge(id) {
        return state.edges.find((edge) => edge.id === id);
      }

      function clearSelection() {
        state.selectedNodeId = null;
        state.selectedEdgeId = null;
      }

      function renderStaticFrame() {
        clearSvg(svg);
        svg.appendChild(svgEl("rect", { x: 0, y: 0, width, height, fill: "#ffffff" }));

        const title = svgEl("text", {
          x: 28,
          y: 34,
          fill: "#0f172a",
          "font-size": 24,
          "font-weight": 700,
        });
        title.textContent = "Force-Directed GRN Network";
        svg.appendChild(title);

        const subtitle = svgEl("text", {
          x: 28,
          y: 58,
          fill: "#475569",
          "font-size": 13.5,
        });
        subtitle.textContent = "Drag nodes to pin them, click to select, and use Re-layout to restart the simulation from the initial random state.";
        svg.appendChild(subtitle);

        svg.appendChild(svgEl("rect", {
          x: 18,
          y: 74,
          width: width - 36,
          height: height - 92,
          rx: 18,
          fill: "#ffffff",
          stroke: "#e2e8f0",
        }));

        const helper = svgEl("text", {
          x: width - 26,
          y: 58,
          fill: "#64748b",
          "font-size": 12.5,
          "text-anchor": "end",
        });
        helper.textContent = "Hover edges for r/support. Dragging a node pins it.";
        svg.appendChild(helper);

        state.refs.edgeLayer = svgEl("g");
        state.refs.nodeLayer = svgEl("g");
        state.refs.legendLayer = svgEl("g");
        svg.appendChild(state.refs.edgeLayer);
        svg.appendChild(state.refs.nodeLayer);
        svg.appendChild(state.refs.legendLayer);
      }

      function updateLegend() {
        const legendLayer = state.refs.legendLayer;
        clearSvg(legendLayer);

        const legendItems = Array.isArray(payload.moduleLegend) ? payload.moduleLegend : [];
        if (!legendItems.length) {
          return;
        }

        const boxWidth = 176;
        const rowHeight = 18;
        const boxHeight = 22 + rowHeight * legendItems.length + 10;
        const boxX = 24;
        const boxY = height - boxHeight - 24;

        legendLayer.appendChild(svgEl("rect", {
          x: boxX,
          y: boxY,
          width: boxWidth,
          height: boxHeight,
          rx: 14,
          fill: "#ffffff",
          stroke: "#dbe4f0",
          opacity: 0.96,
        }));

        const heading = svgEl("text", {
          x: boxX + 14,
          y: boxY + 18,
          fill: "#0f172a",
          "font-size": 12.5,
          "font-weight": 700,
        });
        heading.textContent = "WGCNA modules";
        legendLayer.appendChild(heading);

        legendItems.forEach((item, index) => {
          const y = boxY + 40 + index * rowHeight;
          legendLayer.appendChild(svgEl("rect", {
            x: boxX + 14,
            y: y - 9,
            width: 12,
            height: 12,
            rx: 3,
            fill: item.color,
            stroke: "rgba(15,23,42,0.15)",
          }));
          const label = svgEl("text", {
            x: boxX + 34,
            y,
            fill: "#334155",
            "font-size": 12,
          });
          label.textContent = item.label;
          legendLayer.appendChild(label);
        });
      }

      function edgeDisplayState(edge) {
        const selected = state.selectedEdgeId === edge.id;
        const hovered = !!edge.hovered;
        return {
          stroke: selected ? "#f59e0b" : edge.color,
          strokeWidth: selected ? Number(edge.width) + 2.0 : hovered ? Number(edge.width) + 1.3 : Number(edge.width),
          opacity: selected ? 0.98 : hovered ? Math.min(0.98, Number(edge.opacity) + 0.22) : edge.opacity,
        };
      }

      function nodeDisplayState(node) {
        const selected = state.selectedNodeId === node.id;
        return {
          stroke: selected ? "#f59e0b" : "#ffffff",
          strokeWidth: selected ? 3.0 : 1.8,
        };
      }

      function buildGraph() {
        renderStaticFrame();
        updateLegend();

        activeEdges().forEach((edge) => {
          const line = svgEl("line", {
            x1: 0,
            y1: 0,
            x2: 0,
            y2: 0,
            stroke: edge.color,
            "stroke-width": edge.width,
            opacity: edge.opacity,
            "stroke-linecap": "round",
          });

          line.addEventListener("mouseenter", (event) => {
            edge.hovered = true;
            updateGraphPositions();
            const source = getNode(edge.source);
            const target = getNode(edge.target);
            showTooltip(
              `${source ? source.label : edge.source} → ${target ? target.label : edge.target}<br/>r=${Number(edge.correlation).toFixed(2)}, support=${edge.support}`,
              event.clientX,
              event.clientY
            );
          });
          line.addEventListener("mousemove", (event) => {
            moveTooltip(event.clientX, event.clientY);
          });
          line.addEventListener("mouseleave", () => {
            edge.hovered = false;
            updateGraphPositions();
            hideTooltip();
          });
          line.addEventListener("click", (event) => {
            event.stopPropagation();
            if (state.selectedEdgeId === edge.id) {
              state.selectedEdgeId = null;
            } else {
              state.selectedEdgeId = edge.id;
              state.selectedNodeId = null;
            }
            updateGraphPositions();
          });

          edge.el = line;
          state.refs.edgeLayer.appendChild(line);
        });

        activeNodes().forEach((node) => {
          const group = svgEl("g");
          const shape = node.type === "Gene"
            ? svgEl("circle", { cx: node.x, cy: node.y, r: 8, fill: node.color })
            : svgEl("rect", { x: node.x - 7, y: node.y - 7, width: 14, height: 14, rx: 3, fill: "#111827" });

          const label = svgEl("text", {
            x: node.type === "Gene" ? node.x - 12 : node.x + 12,
            y: node.y,
            fill: "#111827",
            "font-size": 9,
            "font-weight": node.type === "Metabolite" ? 700 : 600,
            "text-anchor": node.type === "Gene" ? "end" : "start",
            "dominant-baseline": "middle",
            opacity: state.showLabelsByDefault ? 1 : 0,
          });
          label.textContent = node.label;

          group.appendChild(shape);
          group.appendChild(label);
          state.refs.nodeLayer.appendChild(group);

          const selectNode = (event) => {
            event.stopPropagation();
            if (state.selectedNodeId === node.id) {
              state.selectedNodeId = null;
            } else {
              state.selectedNodeId = node.id;
              state.selectedEdgeId = null;
            }
            updateGraphPositions();
          };

          group.addEventListener("click", selectNode);

          group.addEventListener("mouseenter", (event) => {
            node.hovered = true;
            label.setAttribute("opacity", "1");
            const moduleLabel = node.type === "Gene" ? node.module : "Metabolite";
            showTooltip(
              `类型: ${node.type} | 名称: ${node.label} | 模块: ${moduleLabel}`,
              event.clientX,
              event.clientY
            );
            updateGraphPositions();
          });
          group.addEventListener("mousemove", (event) => {
            moveTooltip(event.clientX, event.clientY);
          });
          group.addEventListener("mouseleave", () => {
            node.hovered = false;
            if (!state.showLabelsByDefault) {
              label.setAttribute("opacity", "0");
            }
            hideTooltip();
            updateGraphPositions();
          });

          attachSvgDrag(group, svg, {
            start: (event) => {
              state.selectedNodeId = node.id;
              state.selectedEdgeId = null;
              node.dragging = true;
              node.pinned = true;
              node.vx = 0;
              node.vy = 0;
              const current = clientToSvg(svg, event.clientX, event.clientY);
              node.x = clamp(current.x, 20, width - 20);
              node.y = clamp(current.y, 20, height - 20);
              updateGraphPositions();
            },
            move: (moveEvent, delta) => {
              const current = clientToSvg(svg, moveEvent.clientX, moveEvent.clientY);
              node.x = clamp(current.x, 20, width - 20);
              node.y = clamp(current.y, 20, height - 20);
              node.vx = 0;
              node.vy = 0;
              updateGraphPositions();
              showTooltip(
                `类型: ${node.type} | 名称: ${node.label} | 模块: ${node.type === "Gene" ? node.module : "Metabolite"}`,
                moveEvent.clientX,
                moveEvent.clientY
              );
            },
            end: () => {
              node.dragging = false;
              node.pinned = true;
              updateGraphPositions();
              setStatus(`${node.label} pinned at the current position.`);
            },
          });

          node.groupEl = group;
          node.shapeEl = shape;
          node.labelEl = label;
        });

        updateGraphPositions();
      }

      function updateGraphPositions() {
        activeEdges().forEach((edge) => {
          const source = getNode(edge.source);
          const target = getNode(edge.target);
          if (!source || !target || !edge.el) {
            return;
          }
          edge.el.setAttribute("x1", String(source.x));
          edge.el.setAttribute("y1", String(source.y));
          edge.el.setAttribute("x2", String(target.x));
          edge.el.setAttribute("y2", String(target.y));
          const style = edgeDisplayState(edge);
          edge.el.setAttribute("stroke", style.stroke);
          edge.el.setAttribute("stroke-width", String(style.strokeWidth));
          edge.el.setAttribute("opacity", String(style.opacity));
        });

        activeNodes().forEach((node) => {
          if (!node.shapeEl || !node.labelEl) {
            return;
          }
          const style = nodeDisplayState(node);
          if (node.type === "Gene") {
            node.shapeEl.setAttribute("cx", String(node.x));
            node.shapeEl.setAttribute("cy", String(node.y));
          } else {
            node.shapeEl.setAttribute("x", String(node.x - 7));
            node.shapeEl.setAttribute("y", String(node.y - 7));
          }
          node.shapeEl.setAttribute("stroke", style.stroke);
          node.shapeEl.setAttribute("stroke-width", String(style.strokeWidth));

          node.labelEl.setAttribute("x", String(node.type === "Gene" ? node.x - 12 : node.x + 12));
          node.labelEl.setAttribute("y", String(node.y));
          node.labelEl.setAttribute("text-anchor", node.type === "Gene" ? "end" : "start");
          if (state.showLabelsByDefault || node.hovered || state.selectedNodeId === node.id) {
            node.labelEl.setAttribute("opacity", "1");
          } else {
            node.labelEl.setAttribute("opacity", "0");
          }
        });
      }

      function stepSimulation() {
        if (!state.running) {
          return;
        }
        state.iteration += 1;
        const nodes = activeNodes();
        const edges = activeEdges();
        let totalMotion = 0;

        nodes.forEach((node) => {
          node.fx = 0;
          node.fy = 0;
        });

        for (let i = 0; i < nodes.length; i += 1) {
          const a = nodes[i];
          for (let j = i + 1; j < nodes.length; j += 1) {
            const b = nodes[j];
            let dx = b.x - a.x;
            let dy = b.y - a.y;
            let distSq = dx * dx + dy * dy;
            if (distSq < 16) {
              dx = (Math.random() - 0.5) * 4;
              dy = (Math.random() - 0.5) * 4;
              distSq = dx * dx + dy * dy + 1e-3;
            }
            const dist = Math.sqrt(distSq);
            const force = state.repulsionStrength / distSq;
            const fx = (force * dx) / dist;
            const fy = (force * dy) / dist;
            a.fx -= fx;
            a.fy -= fy;
            b.fx += fx;
            b.fy += fy;
          }
        }

        edges.forEach((edge) => {
          const source = getNode(edge.source);
          const target = getNode(edge.target);
          if (!source || !target) {
            return;
          }
          let dx = target.x - source.x;
          let dy = target.y - source.y;
          const dist = Math.max(1e-3, Math.sqrt(dx * dx + dy * dy));
          const spring = state.springStrength * (dist - state.restLength);
          const fx = (spring * dx) / dist;
          const fy = (spring * dy) / dist;
          source.fx += fx;
          source.fy += fy;
          target.fx -= fx;
          target.fy -= fy;
        });

        const centerX = width / 2;
        const centerY = height / 2;

        nodes.forEach((node) => {
          if (node.dragging || node.pinned) {
            node.vx = 0;
            node.vy = 0;
            return;
          }

          node.fx += (centerX - node.x) * state.centerGravity;
          node.fy += (centerY - node.y) * state.centerGravity;

          node.vx = (node.vx + node.fx) * state.damping;
          node.vy = (node.vy + node.fy) * state.damping;
          node.x = clamp(node.x + node.vx, 18, width - 18);
          node.y = clamp(node.y + node.vy, 18, height - 18);

          totalMotion += Math.abs(node.vx) + Math.abs(node.vy);
        });

        updateGraphPositions();

        if (state.iteration >= state.maxIterations || totalMotion < state.moveThreshold) {
          state.running = false;
          state.rafId = null;
          setStatus(`Layout settled after ${state.iteration} iterations.`);
          return;
        }

        state.rafId = requestAnimationFrame(stepSimulation);
      }

      function startSimulation() {
        if (state.rafId) {
          cancelAnimationFrame(state.rafId);
          state.rafId = null;
        }
        state.running = true;
        state.iteration = 0;
        state.rafId = requestAnimationFrame(stepSimulation);
      }

      function deleteSelected() {
        if (state.selectedNodeId) {
          const node = getNode(state.selectedNodeId);
          if (node) {
            node.deleted = true;
            state.edges.forEach((edge) => {
              if (edge.source === node.id || edge.target === node.id) {
                edge.deleted = true;
              }
            });
            clearSelection();
            buildGraph();
            startSimulation();
            setStatus(`Removed ${node.label} and its connected edges from the current view.`);
            return;
          }
        }

        if (state.selectedEdgeId) {
          const edge = getEdge(state.selectedEdgeId);
          if (edge) {
            edge.deleted = true;
            const source = getNode(edge.source);
            const target = getNode(edge.target);
            clearSelection();
            buildGraph();
            startSimulation();
            setStatus(`Removed edge ${source ? source.label : edge.source} → ${target ? target.label : edge.target}.`);
            return;
          }
        }

        setStatus("Select a node or edge first.");
      }

      function relayout() {
        state.nodes.forEach((node) => {
          node.x = Number(node.initialX);
          node.y = Number(node.initialY);
          node.vx = 0;
          node.vy = 0;
          node.dragging = false;
          node.pinned = false;
          node.hovered = false;
        });
        clearSelection();
        buildGraph();
        startSimulation();
        setStatus("Returned nodes to their initial random positions and restarted the force layout.");
      }

      svg.onclick = () => {
        clearSelection();
        updateGraphPositions();
      };

      relayoutBtn.addEventListener("click", relayout);
      deleteBtn.addEventListener("click", deleteSelected);
      svgBtn.addEventListener("click", () => {
        exportSvg(svg, "deepomics_grn_network.svg");
        setStatus("Saved current GRN network as SVG.");
      });
      pngBtn.addEventListener("click", () => {
        exportPng(svg, "deepomics_grn_network.png");
        setStatus("Saved current GRN network as PNG.");
      });

      buildGraph();
      startSimulation();
      setStatus("Running force layout. Drag any node to pin it in place.");
    }

    function initCircleEditor(payload) {
      const svg = document.getElementById("circleSvg");
      const fallback = document.getElementById("circleFallback");
      const statusEl = document.getElementById("circleStatus");
      const setStatus = (message) => {
        statusEl.textContent = message || "";
      };

      if (!payload || !Array.isArray(payload.items) || payload.items.length === 0) {
        fallback.hidden = false;
        svg.hidden = true;
        return;
      }

      const base = { width: payload.width || 940, height: payload.height || 780, cx: 380, cy: 400, radius: 250 };
      svg.setAttribute("viewBox", `0 0 ${base.width} ${base.height}`);

      const state = {
        originalItems: deepCopy(payload.items),
        items: deepCopy(payload.items),
        notes: [],
        selected: null,
        nextNoteId: 1,
        refs: { items: {}, notes: {} },
      };

      function getItem(id) {
        return state.items.find((item) => item.id === id && !item.deleted);
      }

      function getNote(id) {
        return state.notes.find((note) => note.id === id);
      }

      function selectedRecord() {
        if (!state.selected) {
          return null;
        }
        if (state.selected.kind === "item") {
          return getItem(state.selected.id);
        }
        if (state.selected.kind === "note") {
          return getNote(state.selected.id);
        }
        return null;
      }

      function renderNote(note, overlayLayer) {
        const group = svgEl("g", {});
        const estimatedWidth = Math.max(90, 9 * String(note.text).length + 18);
        const isSelected = state.selected && state.selected.kind === "note" && state.selected.id === note.id;

        const rect = svgEl("rect", {
          x: note.x,
          y: note.y - 18,
          width: estimatedWidth,
          height: 28,
          rx: 8,
          fill: isSelected ? "#dbeafe" : "#ffffff",
          stroke: isSelected ? "#2563eb" : "#cbd5e1",
          "stroke-width": isSelected ? 2 : 1.2,
          opacity: 0.96,
        });
        const text = svgEl("text", {
          x: note.x + 10,
          y: note.y,
          fill: "#0f172a",
          "font-size": 13,
          "font-weight": 600,
        });
        text.textContent = note.text;

        group.appendChild(rect);
        group.appendChild(text);
        overlayLayer.appendChild(group);

        const update = () => {
          rect.setAttribute("x", String(note.x));
          rect.setAttribute("y", String(note.y - 18));
          text.setAttribute("x", String(note.x + 10));
          text.setAttribute("y", String(note.y));
        };

        group.addEventListener("click", (event) => {
          event.stopPropagation();
          state.selected = { kind: "note", id: note.id };
          render();
        });

        group.addEventListener("dblclick", (event) => {
          event.stopPropagation();
          const replacement = prompt("Rename annotation", note.text);
          if (replacement !== null && replacement.trim()) {
            note.text = replacement.trim();
            render();
            setStatus("Annotation updated.");
          }
        });

        attachSvgDrag(group, svg, {
          start: () => {
            state.selected = { kind: "note", id: note.id };
          },
          move: (_event, delta) => {
            note.x += delta.dx;
            note.y += delta.dy;
            update();
          },
          end: () => {
            render();
            setStatus("Annotation moved.");
          },
        });
      }

      function renderItem(item, itemLayer) {
        const isSelected = state.selected && state.selected.kind === "item" && state.selected.id === item.id;
        const group = svgEl("g", {});
        itemLayer.appendChild(group);

        const endpoint = {
          x: base.cx + item.x * base.radius,
          y: base.cy - item.y * base.radius,
        };

        const leader = svgEl("line", {
          x1: endpoint.x,
          y1: endpoint.y,
          x2: endpoint.x + item.labelDx,
          y2: endpoint.y + item.labelDy,
          stroke: item.color,
          "stroke-width": 0.8,
          opacity: 0.35,
        });
        const arrow = svgEl("line", {
          x1: base.cx,
          y1: base.cy,
          x2: endpoint.x,
          y2: endpoint.y,
          stroke: item.color,
          "stroke-width": isSelected ? 2.4 : 1.4,
          opacity: 0.88,
          "marker-end": item.type === "Gene" ? "url(#circleArrowGene)" : "url(#circleArrowMetab)",
        });
        const point = svgEl("circle", {
          cx: endpoint.x,
          cy: endpoint.y,
          r: isSelected ? 7 : 5.5,
          fill: item.color,
          stroke: isSelected ? "#111827" : "#ffffff",
          "stroke-width": isSelected ? 2 : 1.4,
        });
        const label = svgEl("text", {
          x: endpoint.x + item.labelDx,
          y: endpoint.y + item.labelDy,
          fill: item.color,
          "font-size": 12.5,
          "font-weight": item.type === "Metabolite" ? 700 : 600,
          "text-anchor": item.labelDx >= 0 ? "start" : "end",
          "dominant-baseline": "middle",
        });
        label.textContent = item.label;

        group.appendChild(leader);
        group.appendChild(arrow);
        group.appendChild(point);
        group.appendChild(label);

        const update = () => {
          const x = base.cx + item.x * base.radius;
          const y = base.cy - item.y * base.radius;
          arrow.setAttribute("x2", String(x));
          arrow.setAttribute("y2", String(y));
          point.setAttribute("cx", String(x));
          point.setAttribute("cy", String(y));
          leader.setAttribute("x1", String(x));
          leader.setAttribute("y1", String(y));
          leader.setAttribute("x2", String(x + item.labelDx));
          leader.setAttribute("y2", String(y + item.labelDy));
          label.setAttribute("x", String(x + item.labelDx));
          label.setAttribute("y", String(y + item.labelDy));
          label.setAttribute("text-anchor", item.labelDx >= 0 ? "start" : "end");
        };

        const selectThis = (event) => {
          event.stopPropagation();
          state.selected = { kind: "item", id: item.id };
          render();
        };

        point.addEventListener("click", selectThis);
        label.addEventListener("click", selectThis);

        point.addEventListener("dblclick", (event) => {
          event.stopPropagation();
          const replacement = prompt("Rename feature", item.label);
          if (replacement !== null && replacement.trim()) {
            item.label = replacement.trim();
            render();
            setStatus("Feature label updated.");
          }
        });
        label.addEventListener("dblclick", (event) => {
          event.stopPropagation();
          const replacement = prompt("Rename feature", item.label);
          if (replacement !== null && replacement.trim()) {
            item.label = replacement.trim();
            render();
            setStatus("Feature label updated.");
          }
        });

        attachSvgDrag(point, svg, {
          start: () => {
            state.selected = { kind: "item", id: item.id };
          },
          move: (_event, delta) => {
            item.x = Math.max(-1.1, Math.min(1.1, item.x + delta.dx / base.radius));
            item.y = Math.max(-1.1, Math.min(1.1, item.y - delta.dy / base.radius));
            update();
          },
          end: () => {
            render();
            setStatus("Feature endpoint moved.");
          },
        });

        attachSvgDrag(label, svg, {
          start: () => {
            state.selected = { kind: "item", id: item.id };
          },
          move: (_event, delta) => {
            item.labelDx += delta.dx;
            item.labelDy += delta.dy;
            update();
          },
          end: () => {
            render();
            setStatus("Feature label moved.");
          },
        });
      }

      function render() {
        clearSvg(svg);

        const defs = svgEl("defs");
        const arrowGene = svgEl("marker", {
          id: "circleArrowGene",
          viewBox: "0 0 10 10",
          refX: 9,
          refY: 5,
          markerWidth: 6,
          markerHeight: 6,
          orient: "auto-start-reverse",
        });
        arrowGene.appendChild(svgEl("path", { d: "M 0 0 L 10 5 L 0 10 z", fill: "#2563eb" }));
        const arrowMetab = svgEl("marker", {
          id: "circleArrowMetab",
          viewBox: "0 0 10 10",
          refX: 9,
          refY: 5,
          markerWidth: 6,
          markerHeight: 6,
          orient: "auto-start-reverse",
        });
        arrowMetab.appendChild(svgEl("path", { d: "M 0 0 L 10 5 L 0 10 z", fill: "#dc2626" }));
        defs.appendChild(arrowGene);
        defs.appendChild(arrowMetab);
        svg.appendChild(defs);

        svg.appendChild(svgEl("rect", { x: 0, y: 0, width: base.width, height: base.height, fill: "#ffffff" }));

        const title = svgEl("text", {
          x: 36,
          y: 40,
          fill: "#0f172a",
          "font-size": 24,
          "font-weight": 700,
        });
        title.textContent = payload.title;
        svg.appendChild(title);

        const subtitle = svgEl("text", {
          x: 36,
          y: 64,
          fill: "#475569",
          "font-size": 13.5,
        });
        subtitle.textContent = payload.subtitle;
        svg.appendChild(subtitle);

        svg.appendChild(svgEl("line", { x1: 100, y1: base.cy, x2: 660, y2: base.cy, stroke: "#cbd5e1", "stroke-width": 1 }));
        svg.appendChild(svgEl("line", { x1: base.cx, y1: 120, x2: base.cx, y2: 680, stroke: "#cbd5e1", "stroke-width": 1 }));
        svg.appendChild(svgEl("circle", {
          cx: base.cx,
          cy: base.cy,
          r: base.radius,
          fill: "none",
          stroke: "#94a3b8",
          "stroke-dasharray": "6 5",
          "stroke-width": 1.3,
        }));

        const xLabel = svgEl("text", {
          x: 680,
          y: base.cy - 12,
          fill: "#334155",
          "font-size": 13,
          "font-weight": 600,
        });
        xLabel.textContent = payload.xLabel;
        svg.appendChild(xLabel);

        const yLabel = svgEl("text", {
          x: base.cx + 14,
          y: 122,
          fill: "#334155",
          "font-size": 13,
          "font-weight": 600,
        });
        yLabel.textContent = payload.yLabel;
        svg.appendChild(yLabel);

        const guide = svgEl("text", {
          x: 700,
          y: 150,
          fill: "#64748b",
          "font-size": 13,
        });
        guide.textContent = "Tip: drag points to move vectors; drag labels separately.";
        svg.appendChild(guide);

        const itemLayer = svgEl("g");
        const overlayLayer = svgEl("g");
        svg.appendChild(itemLayer);
        svg.appendChild(overlayLayer);

        state.items.filter((item) => !item.deleted).forEach((item) => renderItem(item, itemLayer));
        state.notes.forEach((note) => renderNote(note, overlayLayer));
      }

      svg.addEventListener("click", () => {
        state.selected = null;
        render();
      });

      document.getElementById("circleRenameBtn").addEventListener("click", () => {
        const record = selectedRecord();
        if (!record || !state.selected || state.selected.kind !== "item") {
          setStatus("Select a feature label first.");
          return;
        }
        const replacement = prompt("Rename feature", record.label);
        if (replacement !== null && replacement.trim()) {
          record.label = replacement.trim();
          render();
          setStatus("Feature label updated.");
        }
      });

      document.getElementById("circleAddNoteBtn").addEventListener("click", () => {
        const text = prompt("Annotation text", "Note");
        if (text && text.trim()) {
          state.notes.push({
            id: `note_${state.nextNoteId++}`,
            text: text.trim(),
            x: 680,
            y: 90 + 34 * state.notes.length,
          });
          state.selected = { kind: "note", id: state.notes[state.notes.length - 1].id };
          render();
          setStatus("Added a draggable annotation.");
        }
      });

      document.getElementById("circleDeleteBtn").addEventListener("click", () => {
        if (!state.selected) {
          setStatus("Select a feature or note first.");
          return;
        }
        if (state.selected.kind === "item") {
          const item = getItem(state.selected.id);
          if (item) {
            item.deleted = true;
          }
          state.selected = null;
          render();
          setStatus("Selected feature removed from the current figure view.");
          return;
        }
        if (state.selected.kind === "note") {
          state.notes = state.notes.filter((note) => note.id !== state.selected.id);
          state.selected = null;
          render();
          setStatus("Selected annotation removed.");
        }
      });

      document.getElementById("circleResetBtn").addEventListener("click", () => {
        state.items = deepCopy(state.originalItems);
        state.notes = [];
        state.nextNoteId = 1;
        state.selected = null;
        render();
        setStatus("Correlation circle layout reset to the package-generated state.");
      });

      document.getElementById("circleSvgBtn").addEventListener("click", () => {
        exportSvg(svg, "deepomics_correlation_circle_edited.svg");
        setStatus("Saved edited correlation circle as SVG.");
      });

      document.getElementById("circlePngBtn").addEventListener("click", () => {
        exportPng(svg, "deepomics_correlation_circle_edited.png");
        setStatus("Saved edited correlation circle as PNG.");
      });

      render();
      setStatus("Ready. Click or drag any feature to start editing.");
    }

    fillSummary();
    setupTabs();
    initPcaScatter(txPcaPayload, {
      svgId: "txPcaSvg",
      fallbackId: "txPcaFallback",
      statusId: "txPcaStatus",
      colorId: "txPcaColor",
      applyBtnId: "txPcaApplyColor",
      svgBtnId: "txPcaSvgBtn",
      pngBtnId: "txPcaPngBtn",
      svgFilename: "deepomics_transcriptome_pca.svg",
      pngFilename: "deepomics_transcriptome_pca.png",
    });
    initPcaScatter(metabPcaPayload, {
      svgId: "metabPcaSvg",
      fallbackId: "metabPcaFallback",
      statusId: "metabPcaStatus",
      colorId: "metabPcaColor",
      applyBtnId: "metabPcaApplyColor",
      svgBtnId: "metabPcaSvgBtn",
      pngBtnId: "metabPcaPngBtn",
      svgFilename: "deepomics_metabolome_pca.svg",
      pngFilename: "deepomics_metabolome_pca.png",
    });
    initNetwork(networkPayload);
    initCircleEditor(circlePayload);
  </script>
</body>
</html>
"""


def generate_interactive_visual_report(engine, cfg, report_path: str | Path) -> None:
    """Generate a standalone interactive HTML report."""
    output_path = Path(report_path)
    safe_mkdir(output_path.parent)

    tx_matrix = np.asarray(engine.adata.X, dtype=np.float32)
    metab_matrix_raw = engine.adata.obsm.get("metabolomics_scaled", engine.adata.obsm.get("metabolomics"))
    if isinstance(metab_matrix_raw, pd.DataFrame):
        metab_matrix = metab_matrix_raw.to_numpy(dtype=np.float32, copy=False)
    elif metab_matrix_raw is None:
        metab_matrix = np.empty((0, 0), dtype=np.float32)
    else:
        metab_matrix = np.asarray(metab_matrix_raw, dtype=np.float32)

    html_text = _interactive_html_template()
    html_text = html_text.replace("__PROJECT_NAME__", html.escape(str(cfg.project_name)))
    html_text = html_text.replace("__SUMMARY_PAYLOAD__", _json_dumps(_build_summary_payload(engine, cfg)))
    html_text = html_text.replace(
        "__TRANSCRIPTOME_PCA_PAYLOAD__",
        _json_dumps(_build_pca_payload(tx_matrix, engine.adata.obs_names.astype(str).tolist(), "Transcriptome PCA", cfg)),
    )
    html_text = html_text.replace(
        "__METABOLOME_PCA_PAYLOAD__",
        _json_dumps(_build_pca_payload(metab_matrix, engine.adata.obs_names.astype(str).tolist(), "Metabolome PCA", cfg)),
    )
    html_text = html_text.replace("__NETWORK_PAYLOAD__", _json_dumps(_build_network_payload(engine, cfg)))
    html_text = html_text.replace("__CIRCLE_PAYLOAD__", _json_dumps(_build_correlation_circle_payload(engine, cfg)))

    output_path.write_text(html_text, encoding="utf-8")
