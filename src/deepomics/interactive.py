
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.decomposition import PCA

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

    rra_df = engine.ml_results.get("key_genes_rra", pd.DataFrame())
    if isinstance(rra_df, pd.DataFrame) and not rra_df.empty:
        gene_candidates = [g for g in rra_df["Gene"].astype(str).tolist() if g in gene_df.columns]
    else:
        gene_candidates = []
    if len(gene_candidates) < top_genes:
        variance_rank = gene_df.var(axis=0).sort_values(ascending=False).index.astype(str).tolist()
        gene_candidates.extend([g for g in variance_rank if g not in gene_candidates])
    selected_genes = gene_candidates[:top_genes]

    summary_df = engine.ml_results.get("metabolite_summary", pd.DataFrame())
    if isinstance(summary_df, pd.DataFrame) and not summary_df.empty:
        summary_df = summary_df.sort_values(["RRA_Genes", "Candidate_Genes_PCC"], ascending=[False, False])
        metabolite_candidates = [m for m in summary_df["Metabolite"].astype(str).tolist() if m in metab_df.columns]
    else:
        metabolite_candidates = []
    if len(metabolite_candidates) < top_metabolites:
        variance_rank = metab_df.var(axis=0).sort_values(ascending=False).index.astype(str).tolist()
        metabolite_candidates.extend([m for m in variance_rank if m not in metabolite_candidates])
    selected_metabs = metabolite_candidates[:top_metabolites]

    return selected_genes, selected_metabs


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

    palette = {"Gene": "#2563eb", "Metabolite": "#dc2626"}
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


def _build_grn_editor_payload(engine, cfg) -> dict[str, Any] | None:
    """Prepare a draggable GRN editor payload for the browser."""
    edge_df = engine.ml_results.get("grn_edges_df")
    if not isinstance(edge_df, pd.DataFrame) or edge_df.empty:
        return None

    ranked = edge_df.assign(AbsPCC=edge_df["PCC_R"].abs()).sort_values(
        ["Support_Count", "In_RRA", "AbsPCC"],
        ascending=[False, False, False],
    )
    top_edges = ranked.head(cfg.circos_top_edges).copy()
    if top_edges.empty:
        return None

    genes = top_edges["Gene"].astype(str).drop_duplicates().tolist()
    metabolites = top_edges["Metabolite"].astype(str).drop_duplicates().tolist()
    if not genes or not metabolites:
        return None

    module_df = engine.wgcna_results.get("Gene_Modules", pd.DataFrame())
    if isinstance(module_df, pd.DataFrame) and not module_df.empty:
        gene_to_module = dict(zip(module_df["Gene"].astype(str), module_df["Module"].astype(str)))
    else:
        gene_to_module = {}
    used_modules = [gene_to_module.get(gene, "Unassigned") for gene in genes]
    module_colors = _module_color_map(used_modules)

    width = 1240
    height = max(860, 120 + 28 * max(len(genes), len(metabolites)))

    def _lane_positions(n_items: int, lane_top: float, lane_bottom: float) -> np.ndarray:
        if n_items <= 1:
            return np.array([(lane_top + lane_bottom) / 2.0], dtype=float)
        return np.linspace(lane_top, lane_bottom, num=n_items, dtype=float)

    gene_y = _lane_positions(len(genes), 140.0, float(height) - 120.0)
    metab_y = _lane_positions(len(metabolites), 140.0, float(height) - 120.0)

    nodes = []
    for gene, y in zip(genes, gene_y):
        module = gene_to_module.get(gene, "Unassigned")
        nodes.append(
            {
                "id": f"gene::{gene}",
                "label": gene,
                "type": "Gene",
                "module": module,
                "x": 260.0,
                "y": float(y),
                "color": module_colors.get(module, "#bdbdbd"),
            }
        )
    for metab, y in zip(metabolites, metab_y):
        nodes.append(
            {
                "id": f"metab::{metab}",
                "label": metab,
                "type": "Metabolite",
                "module": "Metabolite",
                "x": 980.0,
                "y": float(y),
                "color": "#111827",
            }
        )

    node_set = {node["id"] for node in nodes}
    edges = []
    for edge_idx, (_, row) in enumerate(top_edges.reset_index(drop=True).iterrows(), start=1):
        source = f"gene::{str(row['Gene'])}"
        target = f"metab::{str(row['Metabolite'])}"
        if source not in node_set or target not in node_set:
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
                "width": 1.2 + 3.0 * abs_corr,
                "opacity": min(0.92, 0.28 + 0.16 * support),
                "color": "#dc2626" if corr >= 0 else "#2563eb",
                "custom": False,
            }
        )

    module_legend = [
        {"label": module, "color": module_colors[module]}
        for module in sorted(set(used_modules))
        if module in module_colors
    ]

    return {
        "title": "Prioritized GRN Editor",
        "subtitle": "Drag nodes, delete distracting edges, connect two selected nodes, and export the refined network as SVG or PNG.",
        "width": width,
        "height": height,
        "nodes": nodes,
        "edges": edges,
        "moduleLegend": module_legend,
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
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Inter", "Segoe UI", Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }
    .page {
      max-width: 1440px;
      margin: 0 auto;
      padding: 28px;
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
      gap: 10px;
      margin-bottom: 14px;
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
    .canvas-shell {
      background: #ffffff;
      border: 1px solid var(--border);
      border-radius: 18px;
      overflow: hidden;
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
    .help-list {
      margin: 8px 0 0 0;
      padding-left: 18px;
      color: var(--muted);
      font-size: 14px;
    }
    .fallback {
      border: 1px dashed var(--border);
      border-radius: 14px;
      padding: 18px;
      background: #f8fafc;
      color: var(--muted);
    }
    a.inline-link {
      color: #1d4ed8;
      text-decoration: none;
      font-weight: 600;
    }
    a.inline-link:hover {
      text-decoration: underline;
    }
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>DeepOmics Interactive Figure Studio</h1>
      <p>This standalone HTML report keeps the underlying DeepOmics model outputs untouched while exposing an editable figure layer for publication polishing. It is intentionally focused on the most annotation-heavy figures: the multi-omics correlation circle and the prioritized gene-metabolite network.</p>
      <div class="chip-row" id="summaryChips"></div>
      <div class="callout">
        Recommended workflow: keep the package-generated CSV / H5AD / static SVG outputs as the reproducible record, then use this page only for final presentation edits such as label deconfliction, node movement, edge pruning, and figure export.
      </div>
    </section>

    <div class="grid">
      <section class="card" id="circleCard">
        <h2>Correlation circle editor</h2>
        <p class="desc">Best suited for fixing crowded labels in PCA correlation circles. Drag feature endpoints to move vectors, drag labels independently, double-click labels to rename, add free-text notes, and export the edited figure.</p>
        <div class="toolbar">
          <button class="primary" id="circleRenameBtn">Rename selected feature</button>
          <button id="circleAddNoteBtn">Add note</button>
          <button class="warn" id="circleDeleteBtn">Delete selected item</button>
          <button id="circleResetBtn">Reset layout</button>
          <button id="circleSvgBtn">Save SVG</button>
          <button id="circlePngBtn">Save PNG</button>
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

      <section class="card" id="grnCard">
        <h2>Prioritized GRN editor</h2>
        <p class="desc">Best suited for network layout polishing before manuscript figures. Drag nodes, shift-click two nodes to create a custom edge, click an edge to edit support / correlation, delete selected edges or nodes, add notes, and export the edited result.</p>
        <div class="toolbar">
          <button class="primary" id="grnRenameNodeBtn">Rename selected node</button>
          <button id="grnAddEdgeBtn">Connect two selected nodes</button>
          <button id="grnEditEdgeBtn">Edit selected edge</button>
          <button id="grnAddNoteBtn">Add note</button>
          <button class="warn" id="grnDeleteBtn">Delete selection</button>
          <button id="grnResetBtn">Reset layout</button>
          <button id="grnSvgBtn">Save SVG</button>
          <button id="grnPngBtn">Save PNG</button>
        </div>
        <div class="canvas-shell">
          <svg id="grnSvg" viewBox="0 0 1240 860" role="img" aria-label="GRN editor"></svg>
        </div>
        <div class="legend" id="grnLegend"></div>
        <div class="status" id="grnStatus"></div>
        <div class="fallback" id="grnFallback" hidden>No GRN payload was generated. This usually means no prioritized gene-metabolite edges were available in the current run.</div>
      </section>

      <section class="card">
        <h2>Editing guide</h2>
        <p class="desc">These interactions are designed to stay light-weight and browser-native, so the HTML file remains fully standalone and easy to share.</p>
        <ul class="help-list">
          <li><strong>Correlation circle:</strong> drag the endpoint to reposition the arrow; drag the label text separately to remove overlap; delete the currently selected feature or note.</li>
          <li><strong>GRN editor:</strong> drag any node; click an edge to edit its support count and correlation value; shift-click exactly two nodes to create a new edge; deleting a node also removes its incident edges.</li>
          <li><strong>Export:</strong> both editors can save the current edited state as standalone SVG or rasterized PNG.</li>
          <li><strong>Reproducibility:</strong> the page edits only the rendered figure state. Your underlying DeepOmics tables and model outputs remain unchanged.</li>
        </ul>
      </section>
    </div>
  </div>

  <script>
    const summaryPayload = __SUMMARY_PAYLOAD__;
    const circlePayload = __CIRCLE_PAYLOAD__;
    const grnPayload = __GRN_PAYLOAD__;

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

    function fillSummary() {
      const chips = document.getElementById("summaryChips");
      const rows = [
        ["Project", summaryPayload.projectName],
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

    function initGrnEditor(payload) {
      const svg = document.getElementById("grnSvg");
      const fallback = document.getElementById("grnFallback");
      const legend = document.getElementById("grnLegend");
      const statusEl = document.getElementById("grnStatus");
      const setStatus = (message) => {
        statusEl.textContent = message || "";
      };

      if (!payload || !Array.isArray(payload.nodes) || payload.nodes.length === 0) {
        fallback.hidden = false;
        svg.hidden = true;
        return;
      }

      svg.setAttribute("viewBox", `0 0 ${payload.width} ${payload.height}`);

      const state = {
        originalNodes: deepCopy(payload.nodes),
        originalEdges: deepCopy(payload.edges),
        nodes: deepCopy(payload.nodes),
        edges: deepCopy(payload.edges),
        notes: [],
        selectedNodeIds: [],
        selectedEdgeId: null,
        selectedNoteId: null,
        nextNoteId: 1,
        nextEdgeId: 1,
      };

      function getNode(nodeId) {
        return state.nodes.find((node) => node.id === nodeId && !node.deleted);
      }

      function getEdge(edgeId) {
        return state.edges.find((edge) => edge.id === edgeId && !edge.deleted);
      }

      function clearSelection() {
        state.selectedNodeIds = [];
        state.selectedEdgeId = null;
        state.selectedNoteId = null;
      }

      function renderLegend() {
        legend.innerHTML = "";
        const fixedEntries = [
          { label: "Positive association", color: "#dc2626" },
          { label: "Negative association", color: "#2563eb" },
          { label: "Metabolite node", color: "#111827" },
        ];
        fixedEntries.forEach((entry) => {
          const item = document.createElement("span");
          item.className = "legend-item";
          item.innerHTML = `<span class="legend-swatch" style="background:${entry.color}"></span>${entry.label}`;
          legend.appendChild(item);
        });
        (payload.moduleLegend || []).slice(0, 10).forEach((entry) => {
          const item = document.createElement("span");
          item.className = "legend-item";
          item.innerHTML = `<span class="legend-swatch" style="background:${entry.color}"></span>${entry.label}`;
          legend.appendChild(item);
        });
      }

      function updateEdge(edge, path) {
        const source = getNode(edge.source);
        const target = getNode(edge.target);
        if (!path || !source || !target) {
          return;
        }
        const midX = (source.x + target.x) / 2.0;
        const d = `M ${source.x} ${source.y} C ${midX} ${source.y}, ${midX} ${target.y}, ${target.x} ${target.y}`;
        path.setAttribute("d", d);
      }

      function renderNote(note, overlayLayer) {
        const estimatedWidth = Math.max(92, 9 * String(note.text).length + 18);
        const isSelected = state.selectedNoteId === note.id;

        const group = svgEl("g");
        const rect = svgEl("rect", {
          x: note.x,
          y: note.y - 18,
          width: estimatedWidth,
          height: 28,
          rx: 8,
          fill: isSelected ? "#dcfce7" : "#ffffff",
          stroke: isSelected ? "#059669" : "#cbd5e1",
          "stroke-width": isSelected ? 2 : 1.2,
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
          clearSelection();
          state.selectedNoteId = note.id;
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
            clearSelection();
            state.selectedNoteId = note.id;
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

      function renderEdge(edge, edgeLayer) {
        const source = getNode(edge.source);
        const target = getNode(edge.target);
        if (!source || !target) {
          return;
        }
        const isSelected = state.selectedEdgeId === edge.id;
        const midX = (source.x + target.x) / 2.0;
        const d = `M ${source.x} ${source.y} C ${midX} ${source.y}, ${midX} ${target.y}, ${target.x} ${target.y}`;
        const path = svgEl("path", {
          d,
          fill: "none",
          stroke: edge.color,
          "stroke-width": isSelected ? edge.width + 2.0 : edge.width,
          opacity: edge.opacity,
          "stroke-linecap": "round",
          "stroke-dasharray": edge.custom ? "8 5" : undefined,
        });
        const title = svgEl("title");
        title.textContent = `${source.label} -> ${target.label} | r=${Number(edge.correlation).toFixed(2)} | support=${edge.support}`;
        path.appendChild(title);
        edgeLayer.appendChild(path);

        path.addEventListener("click", (event) => {
          event.stopPropagation();
          state.selectedNodeIds = [];
          state.selectedNoteId = null;
          state.selectedEdgeId = edge.id;
          render();
        });

        path.addEventListener("dblclick", (event) => {
          event.stopPropagation();
          state.selectedNodeIds = [];
          state.selectedNoteId = null;
          state.selectedEdgeId = edge.id;
          editSelectedEdge();
        });

        return path;
      }

      function renderNode(node, nodeLayer, edgePaths) {
        const isSelected = state.selectedNodeIds.includes(node.id);
        const group = svgEl("g");
        const circle = svgEl("circle", {
          cx: node.x,
          cy: node.y,
          r: 9,
          fill: node.color,
          stroke: isSelected ? "#0f172a" : "#ffffff",
          "stroke-width": isSelected ? 3 : 1.8,
        });
        const label = svgEl("text", {
          x: node.x + (node.type === "Gene" ? -12 : 12),
          y: node.y,
          fill: "#111827",
          "font-size": 12.5,
          "font-weight": node.type === "Metabolite" ? 700 : 600,
          "text-anchor": node.type === "Gene" ? "end" : "start",
          "dominant-baseline": "middle",
        });
        label.textContent = node.label;
        group.appendChild(circle);
        group.appendChild(label);
        nodeLayer.appendChild(group);

        const update = () => {
          const nodeSelected = state.selectedNodeIds.includes(node.id);
          circle.setAttribute("cx", String(node.x));
          circle.setAttribute("cy", String(node.y));
          circle.setAttribute("stroke", nodeSelected ? "#0f172a" : "#ffffff");
          circle.setAttribute("stroke-width", nodeSelected ? "3" : "1.8");
          label.setAttribute("x", String(node.x + (node.type === "Gene" ? -12 : 12)));
          label.setAttribute("y", String(node.y));
          label.setAttribute("text-anchor", node.type === "Gene" ? "end" : "start");
          label.textContent = node.label;
          state.edges.filter((edge) => !edge.deleted && (edge.source === node.id || edge.target === node.id)).forEach((edge) => updateEdge(edge, edgePaths[edge.id]));
        };

        const selectNode = (event) => {
          event.stopPropagation();
          state.selectedEdgeId = null;
          state.selectedNoteId = null;
          if (event.shiftKey) {
            if (state.selectedNodeIds.includes(node.id)) {
              state.selectedNodeIds = state.selectedNodeIds.filter((value) => value !== node.id);
            } else {
              state.selectedNodeIds = [...state.selectedNodeIds, node.id].slice(-2);
            }
          } else {
            state.selectedNodeIds = [node.id];
          }
          render();
        };

        group.addEventListener("click", selectNode);
        group.addEventListener("dblclick", (event) => {
          event.stopPropagation();
          state.selectedNodeIds = [node.id];
          renameSelectedNode();
        });

        attachSvgDrag(group, svg, {
          start: () => {
            state.selectedEdgeId = null;
            state.selectedNoteId = null;
            if (!state.selectedNodeIds.includes(node.id)) {
              state.selectedNodeIds = [node.id];
            }
          },
          move: (_event, delta) => {
            node.x = Math.max(120, Math.min(payload.width - 120, node.x + delta.dx));
            node.y = Math.max(100, Math.min(payload.height - 90, node.y + delta.dy));
            update();
          },
          end: () => {
            render();
            setStatus("Node moved.");
          },
        });
      }

      function renameSelectedNode() {
        if (state.selectedNodeIds.length !== 1) {
          setStatus("Select exactly one node to rename.");
          return;
        }
        const node = getNode(state.selectedNodeIds[0]);
        if (!node) {
          setStatus("Selected node is no longer available.");
          return;
        }
        const replacement = prompt("Rename node", node.label);
        if (replacement !== null && replacement.trim()) {
          node.label = replacement.trim();
          render();
          setStatus("Node label updated.");
        }
      }

      function editSelectedEdge() {
        const edge = getEdge(state.selectedEdgeId);
        if (!edge) {
          setStatus("Select one edge first.");
          return;
        }
        const corrRaw = prompt("Set correlation value between -1 and 1", String(edge.correlation ?? 0.5));
        if (corrRaw === null) {
          return;
        }
        const corr = Math.max(-1, Math.min(1, Number(corrRaw)));
        const supportRaw = prompt("Set support count", String(edge.support ?? 1));
        if (supportRaw === null) {
          return;
        }
        const support = Math.max(1, Math.round(Number(supportRaw) || 1));
        edge.correlation = corr;
        edge.support = support;
        edge.color = corr >= 0 ? "#dc2626" : "#2563eb";
        edge.width = 1.2 + 3.0 * Math.min(1, Math.abs(corr));
        edge.opacity = Math.min(0.92, 0.28 + 0.16 * support);
        render();
        setStatus("Edge style updated.");
      }

      function addEdgeFromSelection() {
        if (state.selectedNodeIds.length !== 2) {
          setStatus("Shift-click exactly two nodes, then use Connect two selected nodes.");
          return;
        }
        const first = getNode(state.selectedNodeIds[0]);
        const second = getNode(state.selectedNodeIds[1]);
        if (!first || !second) {
          setStatus("The selected nodes are not available.");
          return;
        }
        if (first.type === second.type) {
          setStatus("Custom edges must connect one gene and one metabolite node.");
          return;
        }
        const geneNode = first.type === "Gene" ? first : second;
        const metaboliteNode = first.type === "Metabolite" ? first : second;
        const exists = state.edges.some(
          (edge) => !edge.deleted && edge.source === geneNode.id && edge.target === metaboliteNode.id
        );
        if (exists) {
          setStatus("That edge already exists in the current figure state.");
          return;
        }
        const edgeId = `edge_custom_${state.nextEdgeId++}`;
        state.edges.push({
          id: edgeId,
          source: geneNode.id,
          target: metaboliteNode.id,
          correlation: 0.5,
          support: 1,
          width: 2.2,
          opacity: 0.8,
          color: "#059669",
          custom: true,
        });
        state.selectedEdgeId = edgeId;
        state.selectedNodeIds = [];
        render();
        setStatus("Custom edge added. Click it to edit correlation / support if needed.");
      }

      function deleteSelection() {
        if (state.selectedEdgeId) {
          const edge = getEdge(state.selectedEdgeId);
          if (edge) {
            edge.deleted = true;
          }
          state.selectedEdgeId = null;
          render();
          setStatus("Selected edge removed from the current figure view.");
          return;
        }
        if (state.selectedNoteId) {
          state.notes = state.notes.filter((note) => note.id !== state.selectedNoteId);
          state.selectedNoteId = null;
          render();
          setStatus("Selected annotation removed.");
          return;
        }
        if (state.selectedNodeIds.length > 0) {
          const selected = new Set(state.selectedNodeIds);
          state.nodes.forEach((node) => {
            if (selected.has(node.id)) {
              node.deleted = true;
            }
          });
          state.edges.forEach((edge) => {
            if (selected.has(edge.source) || selected.has(edge.target)) {
              edge.deleted = true;
            }
          });
          state.selectedNodeIds = [];
          render();
          setStatus("Selected node(s) and their incident edges were removed.");
          return;
        }
        setStatus("Select a node, edge, or note first.");
      }

      function reset() {
        state.nodes = deepCopy(state.originalNodes);
        state.edges = deepCopy(state.originalEdges);
        state.notes = [];
        state.selectedNodeIds = [];
        state.selectedEdgeId = null;
        state.selectedNoteId = null;
        state.nextNoteId = 1;
        state.nextEdgeId = 1;
        render();
        setStatus("GRN layout reset to the package-generated state.");
      }

      function render() {
        clearSvg(svg);
        renderLegend();

        svg.appendChild(svgEl("rect", { x: 0, y: 0, width: payload.width, height: payload.height, fill: "#ffffff" }));

        svg.appendChild(svgEl("rect", {
          x: 90,
          y: 110,
          width: 260,
          height: payload.height - 200,
          rx: 18,
          fill: "#f8fafc",
          stroke: "#e2e8f0",
        }));
        svg.appendChild(svgEl("rect", {
          x: payload.width - 350,
          y: 110,
          width: 260,
          height: payload.height - 200,
          rx: 18,
          fill: "#f8fafc",
          stroke: "#e2e8f0",
        }));

        const title = svgEl("text", {
          x: 34,
          y: 40,
          fill: "#0f172a",
          "font-size": 24,
          "font-weight": 700,
        });
        title.textContent = payload.title;
        svg.appendChild(title);

        const subtitle = svgEl("text", {
          x: 34,
          y: 64,
          fill: "#475569",
          "font-size": 13.5,
        });
        subtitle.textContent = payload.subtitle;
        svg.appendChild(subtitle);

        const geneLane = svgEl("text", {
          x: 220,
          y: 98,
          fill: "#334155",
          "font-size": 15,
          "font-weight": 700,
          "text-anchor": "middle",
        });
        geneLane.textContent = "Genes";
        svg.appendChild(geneLane);

        const metabLane = svgEl("text", {
          x: payload.width - 220,
          y: 98,
          fill: "#334155",
          "font-size": 15,
          "font-weight": 700,
          "text-anchor": "middle",
        });
        metabLane.textContent = "Metabolites";
        svg.appendChild(metabLane);

        const helper = svgEl("text", {
          x: payload.width / 2,
          y: payload.height - 28,
          fill: "#64748b",
          "font-size": 13,
          "text-anchor": "middle",
        });
        helper.textContent = "Shift-click two nodes to create a custom edge. Double-click an edge or node to edit.";
        svg.appendChild(helper);

        const edgeLayer = svgEl("g");
        const nodeLayer = svgEl("g");
        const overlayLayer = svgEl("g");
        svg.appendChild(edgeLayer);
        svg.appendChild(nodeLayer);
        svg.appendChild(overlayLayer);

        const edgePaths = {};
        state.edges.filter((edge) => !edge.deleted).forEach((edge) => {
          edgePaths[edge.id] = renderEdge(edge, edgeLayer);
        });
        state.nodes.filter((node) => !node.deleted).forEach((node) => renderNode(node, nodeLayer, edgePaths));
        state.notes.forEach((note) => renderNote(note, overlayLayer));
      }

      svg.addEventListener("click", () => {
        clearSelection();
        render();
      });

      document.getElementById("grnRenameNodeBtn").addEventListener("click", renameSelectedNode);
      document.getElementById("grnEditEdgeBtn").addEventListener("click", editSelectedEdge);
      document.getElementById("grnAddEdgeBtn").addEventListener("click", addEdgeFromSelection);
      document.getElementById("grnDeleteBtn").addEventListener("click", deleteSelection);
      document.getElementById("grnResetBtn").addEventListener("click", reset);

      document.getElementById("grnAddNoteBtn").addEventListener("click", () => {
        const text = prompt("Annotation text", "Mechanistic note");
        if (text && text.trim()) {
          state.notes.push({
            id: `note_${state.nextNoteId++}`,
            text: text.trim(),
            x: payload.width / 2 - 110,
            y: 96 + 32 * state.notes.length,
          });
          clearSelection();
          state.selectedNoteId = state.notes[state.notes.length - 1].id;
          render();
          setStatus("Added a draggable annotation.");
        }
      });

      document.getElementById("grnSvgBtn").addEventListener("click", () => {
        exportSvg(svg, "deepomics_prioritized_grn_edited.svg");
        setStatus("Saved edited GRN as SVG.");
      });

      document.getElementById("grnPngBtn").addEventListener("click", () => {
        exportPng(svg, "deepomics_prioritized_grn_edited.png");
        setStatus("Saved edited GRN as PNG.");
      });

      render();
      setStatus("Ready. Drag nodes, click edges, or shift-click two nodes to add a custom edge.");
    }

    fillSummary();
    initCircleEditor(circlePayload);
    initGrnEditor(grnPayload);
  </script>
</body>
</html>
"""


def generate_interactive_visual_report(engine, cfg, report_path: str | Path) -> None:
    """Generate a standalone interactive HTML figure studio."""
    output_path = Path(report_path)
    safe_mkdir(output_path.parent)

    html_text = _interactive_html_template()
    html_text = html_text.replace("__PROJECT_NAME__", html.escape(str(cfg.project_name)))
    html_text = html_text.replace("__SUMMARY_PAYLOAD__", _json_dumps(_build_summary_payload(engine, cfg)))
    html_text = html_text.replace("__CIRCLE_PAYLOAD__", _json_dumps(_build_correlation_circle_payload(engine, cfg)))
    html_text = html_text.replace("__GRN_PAYLOAD__", _json_dumps(_build_grn_editor_payload(engine, cfg)))

    output_path.write_text(html_text, encoding="utf-8")
