
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .utils import safe_mkdir


PALETTE = {
    "gene": "#2563eb",
    "metabolite": "#111827",
    "edge_positive": "#dc2626",
    "edge_negative": "#2563eb",
}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Index, pd.Series)):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")


def _json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, default=_json_default)


def _build_summary_payload(engine, cfg) -> dict[str, Any]:
    total_df = engine.ml_results.get("total_association_network_df", pd.DataFrame())
    high_df = engine.ml_results.get("high_confidence_network_df", pd.DataFrame())
    return {
        "projectName": str(cfg.project_name),
        "samples": int(engine.adata.n_obs),
        "genes": int(engine.adata.n_vars),
        "metabolites": int(len(engine.adata.uns.get("metabolite_names", []))),
        "totalEdges": int(len(total_df)) if isinstance(total_df, pd.DataFrame) else 0,
        "highConfidenceEdges": int(len(high_df)) if isinstance(high_df, pd.DataFrame) else 0,
    }


def _build_pca_payload(matrix, sample_names, title: str, cfg) -> dict[str, Any] | None:
    values = matrix.to_numpy(dtype=float, copy=False) if isinstance(matrix, pd.DataFrame) else np.asarray(matrix, dtype=float)

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


def _build_network_payload(engine, tier: str, max_edges: int) -> dict[str, Any] | None:
    if tier == "total":
        edge_df = engine.ml_results.get("total_association_network_df", pd.DataFrame())
        title = "Total Association Network"
    else:
        edge_df = engine.ml_results.get("high_confidence_network_df", pd.DataFrame())
        title = "High-Confidence Association Network"

    if not isinstance(edge_df, pd.DataFrame) or edge_df.empty:
        return None

    ranked = edge_df.sort_values(
        ["EdgeWeight", "RRARank", "ModelSupportCount", "ScreenSupportCount"],
        ascending=[False, True, False, False],
        kind="mergesort",
    ).head(max(1, int(max_edges)))
    if ranked.empty:
        return None

    genes = ranked.groupby("Gene")["EdgeWeight"].max().sort_values(ascending=False).index.astype(str).tolist()
    metabolites = (
        ranked.groupby("Metabolite")["EdgeWeight"].max().sort_values(ascending=False).index.astype(str).tolist()
    )
    if not genes or not metabolites:
        return None

    width = 1100
    height = max(700, 26 * max(len(genes), len(metabolites)) + 140)
    gene_x = 250
    metab_x = 850
    gene_y = np.linspace(70, height - 70, num=len(genes))
    metab_y = np.linspace(70, height - 70, num=len(metabolites))

    nodes = [
        {"id": f"gene::{gene}", "label": gene, "type": "Gene", "color": PALETTE["gene"], "x": float(gene_x), "y": float(y)}
        for gene, y in zip(genes, gene_y)
    ]
    nodes.extend(
        {
            "id": f"metab::{metab}",
            "label": metab,
            "type": "Metabolite",
            "color": PALETTE["metabolite"],
            "x": float(metab_x),
            "y": float(y),
        }
        for metab, y in zip(metabolites, metab_y)
    )

    node_ids = {node["id"] for node in nodes}
    edges = []
    for edge_idx, row in enumerate(ranked.reset_index(drop=True).itertuples(index=False), start=1):
        source = f"gene::{str(row.Gene)}"
        target = f"metab::{str(row.Metabolite)}"
        if source not in node_ids or target not in node_ids:
            continue

        edges.append(
            {
                "id": f"edge_{edge_idx:03d}",
                "source": source,
                "target": target,
                "weight": float(row.EdgeWeight),
                "modelSupport": int(row.ModelSupportCount),
                "screenSupport": int(row.ScreenSupportCount),
                "color": PALETTE["edge_positive"] if str(row.Sign) == "positive" else PALETTE["edge_negative"],
                "width": float(0.8 + 4.2 * float(row.EdgeWeight)),
                "opacity": float(
                    min(
                        0.95,
                        0.20
                        + 0.35 * (float(row.ModelSupportCount) / 2.0)
                        + 0.20 * (float(row.ScreenSupportCount) / 3.0),
                    )
                ),
            }
        )

    if not edges:
        return None
    return {"title": title, "nodes": nodes, "edges": edges, "width": width, "height": height}


def _interactive_html_template() -> str:
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
    <p>This standalone browser report provides lightweight offline visualization preview and SVG export.</p>
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
      <h2>Total Association Network</h2>
      <div class="toolbar">
        <button onclick="saveSvg('totalNetSvg','total_association_network.svg')">Save SVG</button>
      </div>
      <svg id="totalNetSvg" viewBox="0 0 1100 900"></svg>
      <div class="legend">Blue nodes are genes, dark nodes are metabolites. Red edges indicate positive associations and blue edges indicate negative associations.</div>
    </div>

    <div class="card">
      <h2>High-Confidence Association Network</h2>
      <div class="toolbar">
        <button onclick="saveSvg('highNetSvg','high_confidence_network.svg')">Save SVG</button>
      </div>
      <svg id="highNetSvg" viewBox="0 0 1100 900"></svg>
      <div class="legend">Edges are ranked by RRA and weighted by correlation strength, model support, and screening support.</div>
    </div>
  </div>

  <script>
    const summaryPayload = __SUMMARY_PAYLOAD__;
    const txPcaPayload = __TRANSCRIPTOME_PCA_PAYLOAD__;
    const metabPcaPayload = __METABOLOME_PCA_PAYLOAD__;
    const totalNetworkPayload = __TOTAL_NETWORK_PAYLOAD__;
    const highNetworkPayload = __HIGH_NETWORK_PAYLOAD__;

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

    function renderNetwork(svgId, payload) {
      const svg = document.getElementById(svgId);
      clearSvg(svg);
      if (!payload) return;

      svg.setAttribute("viewBox", `0 0 ${payload.width} ${payload.height}`);
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

      const title = el("text", { x: payload.width / 2, y: 28, "text-anchor": "middle", "font-size": 18, "font-weight": 700, fill: "#111827" });
      title.textContent = payload.title || "";
      svg.appendChild(title);
    }

    addChip(`Samples: ${summaryPayload.samples}`);
    addChip(`Genes: ${summaryPayload.genes}`);
    addChip(`Metabolites: ${summaryPayload.metabolites}`);
    addChip(`Total edges: ${summaryPayload.totalEdges}`);
    addChip(`High-confidence edges: ${summaryPayload.highConfidenceEdges}`);

    renderPca("txSvg", txPcaPayload);
    renderPca("metSvg", metabPcaPayload);
    renderNetwork("totalNetSvg", totalNetworkPayload);
    renderNetwork("highNetSvg", highNetworkPayload);
  </script>
</body>
</html>
"""


def generate_interactive_visual_report(engine, cfg, report_path: str | Path) -> None:
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
    html_text = html_text.replace(
        "__TOTAL_NETWORK_PAYLOAD__",
        _json_dumps(_build_network_payload(engine, "total", max_edges=cfg.network_plot_top_edges)),
    )
    html_text = html_text.replace(
        "__HIGH_NETWORK_PAYLOAD__",
        _json_dumps(_build_network_payload(engine, "high_confidence", max_edges=cfg.network_plot_top_edges)),
    )

    output_path.write_text(html_text, encoding="utf-8")
