from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

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


def _get_primary_key_gene_df(engine) -> pd.DataFrame:
    """Return the key-gene table for the configured primary strategy."""
    strategy = str(getattr(engine.config, "grn_primary_strategy", "rra")).lower()
    return engine.ml_results.get(f"key_genes_{strategy}", pd.DataFrame())


def _pick_display_features(engine, top_genes: int, top_metabolites: int) -> tuple[list[str], list[str]]:
    """Choose compact feature subsets for the interactive report."""
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
    return {
        "projectName": str(cfg.project_name),
        "samples": int(engine.adata.n_obs),
        "genes": int(engine.adata.n_vars),
        "metabolites": int(len(engine.adata.uns.get("metabolite_names", []))),
        "grnEdges": int(len(grn_edges_df)) if isinstance(grn_edges_df, pd.DataFrame) else 0,
        "primaryStrategy": str(getattr(cfg, "grn_primary_strategy", "rra")).upper(),
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
        "width": 900,
        "height": 620,
        "xLabel": f"PC1 ({var_exp[0]:.1f}%)",
        "yLabel": f"PC2 ({var_exp[1]:.1f}%)",
        "points": [{"name": name, "x": float(x), "y": float(y)} for name, (x, y) in zip(sample_names, coords)],
    }


def _build_correlation_circle_payload(engine, cfg) -> dict[str, Any] | None:
    """Prepare correlation-circle data for the browser."""
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

    items = []
    feature_types = ["Gene"] * len(gene_names) + ["Metabolite"] * len(metabolite_names)
    for idx, (name, feature_type) in enumerate(zip(combined.columns.astype(str), feature_types), start=1):
        items.append(
            {
                "id": f"{feature_type.lower()}_{idx:03d}",
                "label": str(name),
                "type": feature_type,
                "x": float(np.clip(corr_coords[idx - 1, 0], -1.05, 1.05)),
                "y": float(np.clip(corr_coords[idx - 1, 1], -1.05, 1.05)),
                "color": PALETTE["gene"] if feature_type == "Gene" else PALETTE["metabolite"],
            }
        )

    return {
        "title": "Correlation Circle",
        "width": 900,
        "height": 760,
        "xLabel": f"PC1 ({var_exp[0]:.1f}%)",
        "yLabel": f"PC2 ({var_exp[1]:.1f}%)",
        "items": items,
    }


def _build_network_payload(engine, cfg) -> dict[str, Any] | None:
    """Prepare a compact GRN payload for the browser."""
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

    width = 1100
    height = 700
    gene_x = 250
    metab_x = 850
    gene_y = np.linspace(70, height - 70, num=len(genes)) if genes else []
    metab_y = np.linspace(70, height - 70, num=len(metabolites)) if metabolites else []

    nodes = [
        {
            "id": f"gene::{gene}",
            "label": gene,
            "type": "Gene",
            "color": PALETTE["gene"],
            "x": float(gene_x),
            "y": float(y),
        }
        for gene, y in zip(genes, gene_y)
    ]
    nodes.extend(
        {
            "id": f"metab::{metab}",
            "label": metab,
            "type": "Metabolite",
            "color": PALETTE["metabolite_node"],
            "x": float(metab_x),
            "y": float(y),
        }
        for metab, y in zip(metabolites, metab_y)
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

    return {"nodes": nodes, "edges": edges, "width": width, "height": height}


def _interactive_html_template() -> str:
    """Return a standalone interactive report template."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>DeepOmics Interactive Report - __PROJECT_NAME__</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; color: #111827; background: #f8fafc; }
    .hero, .card { background: #ffffff; border: 1px solid #d1d5db; border-radius: 16px; padding: 20px; margin-bottom: 20px; }
    .hero h1 { margin-top: 0; }
    .chips { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 12px; }
    .chip { background: #eef2ff; border: 1px solid #c7d2fe; border-radius: 999px; padding: 6px 12px; color: #3730a3; font-size: 13px; font-weight: 600; }
    .toolbar { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 12px; }
    button { border: 1px solid #cbd5e1; border-radius: 10px; background: #fff; padding: 8px 12px; cursor: pointer; }
    button:hover { border-color: #93c5fd; }
    svg { width: 100%; background: #fff; border: 1px solid #e5e7eb; border-radius: 14px; }
    .grid { display: grid; gap: 20px; }
    .legend { color: #475569; font-size: 14px; margin-top: 10px; }
  </style>
</head>
<body>
  <div class="hero">
    <h1>DeepOmics Interactive Report</h1>
    <p>This lightweight browser report focuses on the retained machine-learning workflow and GRN visualization outputs.</p>
    <div class="chips" id="chips"></div>
  </div>

  <div class="grid">
    <div class="card">
      <h2>Transcriptome PCA</h2>
      <div class="toolbar">
        <button onclick="saveSvg('txSvg','transcriptome_pca.svg')">Save SVG</button>
      </div>
      <svg id="txSvg" viewBox="0 0 900 620"></svg>
    </div>

    <div class="card">
      <h2>Metabolome PCA</h2>
      <div class="toolbar">
        <button onclick="saveSvg('metSvg','metabolome_pca.svg')">Save SVG</button>
      </div>
      <svg id="metSvg" viewBox="0 0 900 620"></svg>
    </div>

    <div class="card">
      <h2>GRN Network</h2>
      <div class="toolbar">
        <button onclick="saveSvg('netSvg','grn_network.svg')">Save SVG</button>
      </div>
      <svg id="netSvg" viewBox="0 0 1100 700"></svg>
      <div class="legend">Blue nodes are genes, dark nodes are metabolites. Red edges indicate positive PCC and blue edges indicate negative PCC.</div>
    </div>

    <div class="card">
      <h2>Correlation Circle</h2>
      <div class="toolbar">
        <button onclick="saveSvg('circleSvg','correlation_circle.svg')">Save SVG</button>
      </div>
      <svg id="circleSvg" viewBox="0 0 900 760"></svg>
    </div>
  </div>

  <script>
    const summaryPayload = __SUMMARY_PAYLOAD__;
    const txPcaPayload = __TRANSCRIPTOME_PCA_PAYLOAD__;
    const metabPcaPayload = __METABOLOME_PCA_PAYLOAD__;
    const networkPayload = __NETWORK_PAYLOAD__;
    const circlePayload = __CIRCLE_PAYLOAD__;

    function el(tag, attrs = {}) {
      const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
      for (const [k, v] of Object.entries(attrs)) {
        if (v !== undefined && v !== null) node.setAttribute(k, String(v));
      }
      return node;
    }

    function clearSvg(svg) {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
    }

    function addChip(text) {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.textContent = text;
      document.getElementById("chips").appendChild(chip);
    }

    function saveSvg(id, filename) {
      const svg = document.getElementById(id);
      const clone = svg.cloneNode(true);
      clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
      const text = new XMLSerializer().serializeToString(clone);
      const blob = new Blob([text], { type: "image/svg+xml;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    }

    function renderPca(svgId, payload) {
      const svg = document.getElementById(svgId);
      clearSvg(svg);
      if (!payload || !Array.isArray(payload.points) || payload.points.length === 0) return;

      const width = payload.width || 900;
      const height = payload.height || 620;
      const margin = { top: 40, right: 30, bottom: 70, left: 80 };
      const plotWidth = width - margin.left - margin.right;
      const plotHeight = height - margin.top - margin.bottom;

      const xs = payload.points.map(p => p.x);
      const ys = payload.points.map(p => p.y);
      const xmin = Math.min(0, ...xs), xmax = Math.max(0, ...xs);
      const ymin = Math.min(0, ...ys), ymax = Math.max(0, ...ys);
      const xspan = Math.max(1e-6, xmax - xmin), yspan = Math.max(1e-6, ymax - ymin);

      const sx = v => margin.left + ((v - (xmin - 0.12 * xspan)) / (xspan * 1.24)) * plotWidth;
      const sy = v => margin.top + (((ymax + 0.12 * yspan) - v) / (yspan * 1.24)) * plotHeight;

      svg.appendChild(el("rect", { x: 0, y: 0, width, height, fill: "#ffffff" }));
      svg.appendChild(el("line", { x1: margin.left, y1: sy(0), x2: width - margin.right, y2: sy(0), stroke: "#cbd5e1", "stroke-dasharray": "6 4" }));
      svg.appendChild(el("line", { x1: sx(0), y1: margin.top, x2: sx(0), y2: height - margin.bottom, stroke: "#cbd5e1", "stroke-dasharray": "6 4" }));

      for (const point of payload.points) {
        svg.appendChild(el("circle", { cx: sx(point.x), cy: sy(point.y), r: 5, fill: "#4c78a8", stroke: "#ffffff", "stroke-width": 1.2 }));
        const label = el("text", { x: sx(point.x) + 7, y: sy(point.y) - 6, "font-size": 10, fill: "#334155" });
        label.textContent = point.name;
        svg.appendChild(label);
      }

      const title = el("text", { x: width / 2, y: 24, "text-anchor": "middle", "font-size": 18, "font-weight": 700, fill: "#111827" });
      title.textContent = payload.title || "";
      svg.appendChild(title);

      const xlabel = el("text", { x: width / 2, y: height - 18, "text-anchor": "middle", "font-size": 13, fill: "#334155" });
      xlabel.textContent = payload.xLabel || "PC1";
      svg.appendChild(xlabel);

      const ylabel = el("text", { x: 22, y: height / 2, transform: `rotate(-90 22 ${height / 2})`, "text-anchor": "middle", "font-size": 13, fill: "#334155" });
      ylabel.textContent = payload.yLabel || "PC2";
      svg.appendChild(ylabel);
    }

    function renderNetwork() {
      const svg = document.getElementById("netSvg");
      clearSvg(svg);
      const payload = networkPayload;
      if (!payload) return;

      const nodeMap = new Map(payload.nodes.map(n => [n.id, n]));
      svg.appendChild(el("rect", { x: 0, y: 0, width: payload.width, height: payload.height, fill: "#ffffff" }));

      for (const edge of payload.edges) {
        const source = nodeMap.get(edge.source);
        const target = nodeMap.get(edge.target);
        if (!source || !target) continue;
        svg.appendChild(el("line", {
          x1: source.x, y1: source.y, x2: target.x, y2: target.y,
          stroke: edge.color, "stroke-width": edge.width, opacity: edge.opacity
        }));
      }

      for (const node of payload.nodes) {
        svg.appendChild(el("circle", { cx: node.x, cy: node.y, r: node.type === "Gene" ? 8 : 9, fill: node.color, stroke: "#ffffff", "stroke-width": 1.2 }));
        const text = el("text", {
          x: node.type === "Gene" ? node.x - 12 : node.x + 12,
          y: node.y + 4,
          "text-anchor": node.type === "Gene" ? "end" : "start",
          "font-size": 11,
          fill: "#334155"
        });
        text.textContent = node.label;
        svg.appendChild(text);
      }
    }

    function renderCircle() {
      const svg = document.getElementById("circleSvg");
      clearSvg(svg);
      const payload = circlePayload;
      if (!payload || !Array.isArray(payload.items) || payload.items.length === 0) return;

      const width = payload.width || 900;
      const height = payload.height || 760;
      const cx = width / 2;
      const cy = height / 2 + 20;
      const radius = Math.min(width, height) * 0.30;

      svg.appendChild(el("rect", { x: 0, y: 0, width, height, fill: "#ffffff" }));
      svg.appendChild(el("circle", { cx, cy, r: radius, fill: "none", stroke: "#94a3b8", "stroke-dasharray": "6 4" }));
      svg.appendChild(el("line", { x1: cx - radius * 1.1, y1: cy, x2: cx + radius * 1.1, y2: cy, stroke: "#cbd5e1" }));
      svg.appendChild(el("line", { x1: cx, y1: cy - radius * 1.1, x2: cx, y2: cy + radius * 1.1, stroke: "#cbd5e1" }));

      for (const item of payload.items) {
        const x = cx + item.x * radius;
        const y = cy - item.y * radius;
        svg.appendChild(el("line", { x1: cx, y1: cy, x2: x, y2: y, stroke: item.color, "stroke-width": 1.2, opacity: 0.8 }));
        svg.appendChild(el("circle", { cx: x, cy: y, r: 4.5, fill: item.color }));
        const text = el("text", { x: x + 8, y: y - 6, "font-size": 11, fill: item.color });
        text.textContent = item.label;
        svg.appendChild(text);
      }

      const title = el("text", { x: width / 2, y: 26, "text-anchor": "middle", "font-size": 18, "font-weight": 700, fill: "#111827" });
      title.textContent = payload.title || "";
      svg.appendChild(title);
    }

    addChip(`Samples: ${summaryPayload.samples}`);
    addChip(`Genes: ${summaryPayload.genes}`);
    addChip(`Metabolites: ${summaryPayload.metabolites}`);
    addChip(`GRN edges: ${summaryPayload.grnEdges}`);
    addChip(`Primary strategy: ${summaryPayload.primaryStrategy}`);

    renderPca("txSvg", txPcaPayload);
    renderPca("metSvg", metabPcaPayload);
    renderNetwork();
    renderCircle();
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
