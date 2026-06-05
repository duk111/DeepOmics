from __future__ import annotations

def _interactive_html_template() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>OmicsPrism Interactive Report - __PROJECT_NAME__</title>
  <style>
    :root {
      --bg: #f6f7fb;
      --panel: #ffffff;
      --border: #d7dde5;
      --border-strong: #b8c1cc;
      --text: #111827;
      --muted: #5b6472;
      --accent: #2563eb;
      --accent-soft: #e8eefc;
      --disabled: #cbd5e1;
      --shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    .app {
      display: grid;
      grid-template-columns: 270px minmax(0, 1fr);
      min-height: 100vh;
    }
    .sidebar {
      border-right: 1px solid var(--border);
      background: #fbfcfe;
      padding: 20px 16px;
    }
    .brand {
      font-size: 20px;
      font-weight: 700;
      line-height: 1.2;
      margin: 0 0 8px 0;
    }
    .subtle {
      margin: 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }
    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 16px 0 18px;
    }
    .chip {
      display: inline-flex;
      align-items: center;
      padding: 6px 10px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: #1e3a8a;
      font-size: 12px;
      font-weight: 700;
      border: 1px solid #c7d2fe;
    }
    .nav {
      display: grid;
      gap: 8px;
      margin-top: 16px;
    }
    .nav button {
      width: 100%;
      border: 1px solid var(--border);
      background: var(--panel);
      color: var(--text);
      border-radius: 10px;
      padding: 10px 12px;
      text-align: left;
      cursor: pointer;
      box-shadow: var(--shadow);
      font-size: 13px;
      line-height: 1.3;
    }
    .nav button.active {
      border-color: #93c5fd;
      background: #eff6ff;
    }
    .nav button.pending {
      color: var(--disabled);
      background: #f8fafc;
      box-shadow: none;
    }
    .main {
      padding: 20px 20px 24px;
      min-width: 0;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      box-shadow: var(--shadow);
    }
    .panel + .panel {
      margin-top: 16px;
    }
    .panel-head {
      padding: 16px 16px 10px;
      border-bottom: 1px solid var(--border);
    }
    .panel-title {
      margin: 0;
      font-size: 18px;
      font-weight: 700;
    }
    .panel-note {
      margin: 6px 0 0;
      color: var(--muted);
      font-size: 13px;
    }
    .controls {
      padding: 14px 16px;
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px 14px;
      border-bottom: 1px solid var(--border);
    }
    .control {
      min-width: 0;
      display: grid;
      gap: 6px;
    }
    .control label {
      font-size: 12px;
      font-weight: 700;
      color: var(--muted);
    }
    .control input[type="number"],
    .control input[type="text"],
    .control input[type="range"],
    .control select {
      width: 100%;
      border: 1px solid var(--border-strong);
      border-radius: 8px;
      background: #fff;
      color: var(--text);
      padding: 8px 10px;
      font-size: 13px;
    }
    .control input[type="range"] {
      padding: 8px 0;
    }
    .toggle-row {
      display: flex;
      align-items: center;
      gap: 8px;
      min-height: 36px;
    }
    .toggle-row input {
      width: 16px;
      height: 16px;
    }
    .action-bar {
      padding: 0 16px 14px;
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }
    .action-bar button {
      border: 1px solid var(--border);
      background: #fff;
      color: var(--text);
      border-radius: 8px;
      padding: 8px 12px;
      font-size: 13px;
      cursor: pointer;
    }
    .action-bar button:hover {
      border-color: #93c5fd;
    }
    .chart-wrap {
      padding: 16px;
      overflow: auto;
    }
    .chart-shell {
      position: relative;
      display: inline-block;
      background: #fff;
      border: 1px solid var(--border);
      border-radius: 12px;
      box-shadow: var(--shadow);
    }
    .gallery-head {
      padding: 16px 16px 10px;
      border-bottom: 1px solid var(--border);
    }
    .gallery-title {
      margin: 0;
      font-size: 18px;
      font-weight: 700;
    }
    .gallery-note {
      margin: 6px 0 0;
      color: var(--muted);
      font-size: 13px;
    }
    .gallery-summary {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      padding: 12px 16px 0;
      color: var(--muted);
      font-size: 12px;
    }
    .figure-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 14px;
      padding: 16px;
    }
    .figure-card {
      border: 1px solid var(--border);
      border-radius: 12px;
      background: #fff;
      box-shadow: var(--shadow);
      overflow: hidden;
      cursor: pointer;
      display: grid;
      gap: 0;
      min-width: 0;
    }
    .figure-card.disabled {
      cursor: default;
      opacity: 0.72;
    }
    .figure-card:hover {
      border-color: #93c5fd;
    }
    .figure-thumb {
      width: 100%;
      aspect-ratio: 4 / 3;
      background: linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%);
      border-bottom: 1px solid var(--border);
      display: grid;
      place-items: center;
      overflow: hidden;
    }
    .figure-thumb img {
      display: block;
      width: 100%;
      height: 100%;
      object-fit: contain;
    }
    .figure-body {
      padding: 12px 12px 14px;
      display: grid;
      gap: 8px;
    }
    .figure-kicker {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
    }
    .figure-name {
      margin: 0;
      font-size: 14px;
      font-weight: 700;
      line-height: 1.35;
    }
    .figure-badge {
      flex: 0 0 auto;
      padding: 3px 8px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: #f8fafc;
      color: #334155;
      font-size: 11px;
      line-height: 1.4;
      white-space: nowrap;
    }
    .figure-description {
      margin: 0;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }
    .figure-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      color: var(--muted);
      font-size: 11px;
    }
    .figure-chip {
      padding: 3px 8px;
      border-radius: 999px;
      background: #f3f4f6;
      border: 1px solid #e5e7eb;
    }
    .detail-panel {
      padding: 16px;
      display: grid;
      gap: 14px;
    }
    .detail-head {
      display: grid;
      gap: 6px;
    }
    .detail-title {
      margin: 0;
      font-size: 18px;
      font-weight: 700;
    }
    .detail-text {
      margin: 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }
    .detail-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .detail-actions a,
    .detail-actions button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      border: 1px solid var(--border);
      background: #fff;
      color: var(--text);
      border-radius: 8px;
      padding: 8px 12px;
      font-size: 13px;
      cursor: pointer;
      text-decoration: none;
    }
    .detail-actions a:hover,
    .detail-actions button:hover {
      border-color: #93c5fd;
    }
    .detail-preview {
      display: grid;
      gap: 10px;
    }
    .detail-preview img {
      display: block;
      width: 100%;
      max-width: 100%;
      border: 1px solid var(--border);
      border-radius: 12px;
      background: #fff;
    }
    .detail-links {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    svg {
      display: block;
      max-width: none;
      background: #fff;
    }
    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 10px 14px;
      padding: 14px 16px 16px;
      border-top: 1px solid var(--border);
      color: var(--muted);
      font-size: 12px;
    }
    .legend-item {
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    .swatch {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      border: 1px solid rgba(15, 23, 42, 0.14);
    }
    .placeholder {
      padding: 24px 16px;
      color: var(--muted);
      font-size: 13px;
    }
    .runtime-error {
      margin: 20px;
      padding: 16px;
      border: 1px solid #fecaca;
      border-radius: 12px;
      background: #fff1f2;
      color: #991b1b;
      font-family: Consolas, monospace;
      white-space: pre-wrap;
    }
    @media (max-width: 1000px) {
      .app { grid-template-columns: 1fr; }
      .sidebar { border-right: 0; border-bottom: 1px solid var(--border); }
      .controls { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 720px) {
      .controls { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div id="app" class="app"></div>
  <script id="omicsprism-payload" type="application/json">__PAYLOAD__</script>
  <script>
    const report = JSON.parse(document.getElementById("omicsprism-payload").textContent);
    const figures = Array.isArray(report.figures) ? report.figures : [];

    const state = {
      activeViewId: report.initial_state.activeViewId,
      activeFigureId: report.initial_state.activeFigureId || "",
      controls: JSON.parse(JSON.stringify(report.initial_state.controls || {}))
    };

    const app = document.getElementById("app");

    function renderRuntimeError(error) {
      const message = error && error.stack ? error.stack : String(error);
      const main = el("main", { className: "main" });
      main.appendChild(el("div", { className: "runtime-error", text: message }));
      return main;
    }

    window.addEventListener("error", event => {
      clear(app);
      app.appendChild(renderSidebar());
      app.appendChild(renderRuntimeError(event.error || event.message));
    });

    function el(tag, attrs = {}, children = []) {
      const node = document.createElement(tag);
      for (const [key, value] of Object.entries(attrs)) {
        if (value === undefined || value === null) continue;
        if (key === "className") {
          node.className = value;
        } else if (key === "checked") {
          node.checked = Boolean(value);
        } else if (key === "selected") {
          node.selected = Boolean(value);
        } else if (key === "text") {
          node.textContent = value;
        } else if (key === "html") {
          node.innerHTML = value;
        } else if (key.startsWith("on") && typeof value === "function") {
          node.addEventListener(key.slice(2).toLowerCase(), value);
        } else {
          node.setAttribute(key, String(value));
        }
      }
      for (const child of children) {
        if (child !== null && child !== undefined) node.appendChild(child);
      }
      return node;
    }

    function svgEl(tag, attrs = {}) {
      const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
      for (const [key, value] of Object.entries(attrs)) {
        if (value !== undefined && value !== null) node.setAttribute(key, String(value));
      }
      return node;
    }

    function clear(node) {
      while (node.firstChild) node.removeChild(node.firstChild);
    }

    function clamp(value, min, max) {
      return Math.max(min, Math.min(max, value));
    }

    function fmtPct(value) {
      return `${Number(value || 0).toFixed(1)}%`;
    }

    function getView(id) {
      return report.views.find(v => v.id === id);
    }

    function getDatasetKeyFromControls() {
      const controls = state.controls.pca || {};
      return controls.dataset === "metabolome" ? "pca.metabolome" : "pca.transcriptome";
    }

    function getActiveDataset() {
      return report.datasets[getDatasetKeyFromControls()] || report.datasets["pca.transcriptome"] || report.datasets["pca.metabolome"] || null;
    }

    function getViewControls(viewId) {
      if (!state.controls[viewId]) state.controls[viewId] = {};
      return state.controls[viewId];
    }

    function getAssociationDataset() {
      const controls = getViewControls("association");
      const pairType = controls.pairType === "module_metabolite" ? "module_metabolite" : "gene_metabolite";
      return report.datasets[`association.${pairType}`] || report.datasets["association.gene_metabolite"] || report.datasets["association.module_metabolite"] || null;
    }

    function getNetworkDataset() {
      return report.datasets["network.high_confidence"] || null;
    }

    function getDatasetForView(viewId) {
      if (viewId === "pca") return getActiveDataset();
      if (viewId === "association") return getAssociationDataset();
      if (viewId === "module_heatmap") return report.datasets.module_heatmap || null;
      if (viewId === "network_explorer") return getNetworkDataset();
      return null;
    }

    function cloneObject(value) {
      return JSON.parse(JSON.stringify(value || {}));
    }

    function getFigure(id) {
      return figures.find(figure => figure.id === id) || null;
    }

    function canOpenFigureStudio(figure) {
      if (!figure || !figure.interactiveViewId) return false;
      const targetView = getView(figure.interactiveViewId);
      return Boolean(targetView && targetView.enabled);
    }

    function applyFigurePreset(figure) {
      if (!figure || !figure.interactiveViewId) return;
      const viewId = figure.interactiveViewId;
      const baseControls = cloneObject(report.initial_state.controls?.[viewId] || {});
      state.controls[viewId] = {
        ...baseControls,
        ...(figure.interactiveControls || {})
      };
      if (viewId === "association") {
        const dataset = getAssociationDataset();
        syncAssociationControlsFromDataset(state.controls[viewId], dataset);
      }
    }

    function openFigurePreview(figure) {
      if (!figure) return;
      state.activeViewId = "gallery";
      state.activeFigureId = figure.id;
      render();
    }

    function openFigureStudio(figure) {
      if (!canOpenFigureStudio(figure)) {
        openFigurePreview(figure);
        return;
      }
      applyFigurePreset(figure);
      state.activeFigureId = "";
      state.activeViewId = figure.interactiveViewId;
      render();
    }

    function openFigure(figure) {
      if (canOpenFigureStudio(figure)) {
        openFigureStudio(figure);
      } else {
        openFigurePreview(figure);
      }
    }

    function findAssociationEdge(dataset, controls) {
      if (!dataset) return null;
      const topEdgeId = String(controls.topEdgeId || "").trim();
      const geneOptions = Array.isArray(dataset.geneOptions) ? dataset.geneOptions : [];
      const metaboliteOptions = Array.isArray(dataset.metaboliteOptions) ? dataset.metaboliteOptions : [];
      let gene = String(controls.gene || "").trim();
      let metabolite = String(controls.metabolite || "").trim();

      let known = null;
      if (topEdgeId && Array.isArray(dataset.topEdges)) {
        known = dataset.topEdges.find(edge => edge.id === topEdgeId) || null;
        if (known) {
          gene = String(known.gene || known.module || gene).trim();
          metabolite = String(known.metabolite || metabolite).trim();
        }
      }
      if (!gene || !dataset.xMatrix || !Object.prototype.hasOwnProperty.call(dataset.xMatrix, gene)) {
        gene = geneOptions.length ? String(geneOptions[0].value) : "";
      }
      if (!metabolite || !dataset.yMatrix || !Object.prototype.hasOwnProperty.call(dataset.yMatrix, metabolite)) {
        metabolite = metaboliteOptions.length ? String(metaboliteOptions[0].value) : "";
      }
      if (!gene || !metabolite || !dataset.xMatrix?.[gene] || !dataset.yMatrix?.[metabolite]) return null;

      const pairPrefix = dataset.kind === "module_metabolite" ? "module" : "gene";
      const pairId = `${pairPrefix}||${gene}||${metabolite}`;
      if (!known && Array.isArray(dataset.topEdges)) known = dataset.topEdges.find(edge => edge.id === pairId) || null;

      const geneInfo = dataset.geneModules?.[gene] || {};
      const moduleName = dataset.kind === "module_metabolite" ? gene : (geneInfo.module || "Unassigned");
      const moduleColor = dataset.kind === "module_metabolite"
        ? (dataset.moduleColors?.[gene] || "#9ca3af")
        : (geneInfo.color || "#4c78a8");
      return {
        ...(known || {}),
        id: pairId,
        kind: dataset.kind,
        gene,
        module: moduleName,
        metabolite,
        label: dataset.kind === "module_metabolite" ? `${gene} module vs ${metabolite}` : `${gene} vs ${metabolite}`,
        xLabel: dataset.kind === "module_metabolite" ? `${gene} module eigengene` : gene,
        yLabel: metabolite,
        moduleColor,
        pointColor: moduleColor,
        rLabel: dataset.kind === "module_metabolite" ? "rho" : "r",
        rValue: null,
        x: dataset.xMatrix[gene],
        y: dataset.yMatrix[metabolite],
      };
    }

    function syncAssociationControlsFromDataset(controls, dataset) {
      if (!dataset) {
        controls.topEdgeId = "";
        controls.gene = "";
        controls.metabolite = "";
        return;
      }

      const current = findAssociationEdge(dataset, controls);
      if (!current) return;
      const known = Array.isArray(dataset.topEdges) ? dataset.topEdges.find(edge => edge.id === current.id) : null;
      controls.topEdgeId = known ? current.id : "";
      controls.gene = current.gene;
      controls.metabolite = current.metabolite;
    }

    function setControl(viewId, key, value) {
      const controls = getViewControls(viewId);
      controls[key] = value;

      if (viewId === "association") {
        const dataset = getAssociationDataset();
        if (key === "pairType") {
          syncAssociationControlsFromDataset(controls, dataset);
        } else if (key === "topEdgeId") {
          const edge = findAssociationEdge(dataset, controls);
          if (edge) {
            controls.topEdgeId = Array.isArray(dataset.topEdges) && dataset.topEdges.find(item => item.id === edge.id) ? edge.id : "";
            controls.gene = edge.gene;
            controls.metabolite = edge.metabolite;
          }
        } else if (key === "gene") {
          controls.topEdgeId = "";
          const edge = findAssociationEdge(dataset, controls);
          controls.topEdgeId = edge && Array.isArray(dataset.topEdges) && dataset.topEdges.find(item => item.id === edge.id) ? edge.id : "";
          if (edge) controls.metabolite = edge.metabolite;
        } else if (key === "metabolite") {
          controls.topEdgeId = "";
          const edge = findAssociationEdge(dataset, controls);
          controls.topEdgeId = edge && Array.isArray(dataset.topEdges) && dataset.topEdges.find(item => item.id === edge.id) ? edge.id : "";
          if (edge) controls.gene = edge.gene;
        }
      } else if (viewId === "network_explorer") {
        if (key !== "selectedNodeId") {
          controls.selectedNodeId = "";
        }
      }

      render();
    }

    function resetControls(viewId) {
      state.controls[viewId] = JSON.parse(JSON.stringify(report.initial_state.controls[viewId] || {}));
      if (viewId === "association") {
        const dataset = getAssociationDataset();
        syncAssociationControlsFromDataset(state.controls[viewId], dataset);
      }
      render();
    }

    function downloadSvg(svgNode, filename) {
      const clone = svgNode.cloneNode(true);
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
      setTimeout(() => URL.revokeObjectURL(url), 1200);
    }

    function renderSidebar() {
      const sidebar = el("aside", { className: "sidebar" });
      sidebar.appendChild(el("h1", { className: "brand", text: report.meta.projectName || "OmicsPrism" }));
      sidebar.appendChild(el("p", { className: "subtle", text: "Offline interactive report" }));

      const chips = el("div", { className: "chips" }, [
        el("span", { className: "chip", text: `Samples: ${report.meta.samples}` }),
        el("span", { className: "chip", text: `Genes: ${report.meta.genes}` }),
        el("span", { className: "chip", text: `Metabolites: ${report.meta.metabolites}` }),
        el("span", { className: "chip", text: `Figures: ${report.meta.figureCount || figures.length}` })
      ]);
      sidebar.appendChild(chips);

      const nav = el("div", { className: "nav" });
      for (const view of report.views) {
        const button = el("button", {
          className: [view.id === state.activeViewId ? "active" : "", !view.enabled ? "pending" : ""].filter(Boolean).join(" "),
          text: view.title,
          onclick: () => {
            state.activeViewId = view.id;
            state.activeFigureId = "";
            render();
          }
        });
        nav.appendChild(button);
      }
      sidebar.appendChild(nav);
      return sidebar;
    }

    function getControlOptions(view, control) {
      if (!control.optionsSource) return control.options || [];
      const dataset = getDatasetForView(view.id);
      if (!dataset) return [];

      let options = [];
      if (control.optionsSource === "topEdges") {
        options = (dataset.topEdges || []).map(edge => ({
          value: edge.id,
          label: edge.label || `${edge.gene} vs ${edge.metabolite}`,
        }));
      } else if (control.optionsSource === "geneOptions") {
        options = dataset.geneOptions || [];
      } else if (control.optionsSource === "metaboliteOptions") {
        options = dataset.metaboliteOptions || [];
      }

      if (control.allowEmpty) {
        options = [{ value: "", label: control.emptyLabel || "None" }, ...options];
      }
      return options;
    }

    function renderControlField(view, control) {
      const current = getViewControls(view.id)[control.id];
      const wrapper = el("div", { className: "control" });
      wrapper.appendChild(el("label", { text: control.label }));

      if (control.type === "toggle") {
        const input = el("input", {
          type: "checkbox",
          onchange: event => setControl(view.id, control.id, event.target.checked)
        });
        input.checked = Boolean(current);
        wrapper.appendChild(el("div", { className: "toggle-row" }, [input, el("span", { text: control.description || "" })]));
        return wrapper;
      }

      if (control.type === "range") {
        const input = el("input", {
          type: "range",
          min: control.min,
          max: control.max,
          step: control.step,
          value: current ?? control.default,
          oninput: event => setControl(view.id, control.id, Number(event.target.value))
        });
        const value = el("span", { text: String(current ?? control.default) });
        const shell = el("div", { className: "toggle-row" }, [input, value]);
        input.addEventListener("input", () => { value.textContent = String(input.value); });
        wrapper.appendChild(shell);
        return wrapper;
      }

      if (control.type === "number") {
        const input = el("input", {
          type: "number",
          min: control.min,
          max: control.max,
          step: control.step,
          value: current ?? control.default,
          oninput: event => {
            const next = Number(event.target.value);
            if (Number.isFinite(next)) setControl(view.id, control.id, next);
          }
        });
        wrapper.appendChild(input);
        return wrapper;
      }

      if (control.type === "text") {
        const input = el("input", {
          type: "text",
          value: current ?? control.default ?? "",
          oninput: event => setControl(view.id, control.id, event.target.value)
        });
        wrapper.appendChild(input);
        return wrapper;
      }

      const select = el("select", {
        onchange: event => setControl(view.id, control.id, event.target.value)
      });
      for (const option of getControlOptions(view, control)) {
        const opt = el("option", {
          value: option.value,
          text: option.label || option.value
        });
        opt.selected = String(current ?? control.default) === String(option.value);
        select.appendChild(opt);
      }
      wrapper.appendChild(select);
      return wrapper;
    }

    function resolveAssociationControlDefaults(dataset, controls) {
      if (!dataset) {
        return;
      }
      syncAssociationControlsFromDataset(controls, dataset);
    }

    function renderPcaLegend(dataset, colorBy) {
      const legend = el("div", { className: "legend" });
      const group1Info = dataset.groupOptions?.group1;
      const group2Info = dataset.groupOptions?.group2;
      if (!group1Info || !Array.isArray(group1Info.order) || group1Info.order.length === 0) {
        legend.appendChild(el("span", { text: "No group legend available." }));
        return legend;
      }
      if (colorBy === "group2" && group2Info && Array.isArray(group2Info.order) && group2Info.order.length > 0) {
        const group2Colors = group2Info.colors || {};
        legend.appendChild(el("span", { className: "legend-item", text: "Color: Group 2" }));
        for (const groupName of group2Info.order) {
          const swatch = el("span", { className: "swatch", style: `background:${group2Colors[groupName] || "#6b7280"}` });
          legend.appendChild(el("span", { className: "legend-item" }, [swatch, el("span", { text: groupName })]));
        }
        legend.appendChild(el("span", { className: "legend-item", text: "Shape: Group 1" }));
        for (const groupName of group1Info.order) {
          const marker = group1Info.markers?.[groupName] || "circle";
          const icon = svgEl("svg", { width: 16, height: 16, viewBox: "0 0 16 16" });
          icon.appendChild(pcaMarkerNode(marker, 8, 8, 5, "#111827", "#ffffff"));
          legend.appendChild(el("span", { className: "legend-item" }, [icon, el("span", { text: groupName })]));
        }
      } else {
        const group1Colors = group1Info.colors || {};
        legend.appendChild(el("span", { className: "legend-item", text: "Color: Group 1" }));
        for (const groupName of group1Info.order) {
          const swatch = el("span", { className: "swatch", style: `background:${group1Colors[groupName] || "#6b7280"}` });
          legend.appendChild(el("span", { className: "legend-item" }, [swatch, el("span", { text: groupName })]));
        }
      }
      return legend;
    }

    function figurePreviewNode(figure) {
      const wrapper = el("div", { className: "figure-thumb" });
      const img = el("img", {
        src: figure.previewPath,
        alt: figure.title,
        loading: "lazy",
        onerror: event => {
          event.target.style.display = "none";
          if (!wrapper.querySelector(".placeholder")) {
            wrapper.appendChild(el("span", { className: "placeholder", text: "Preview unavailable" }));
          }
        }
      });
      wrapper.appendChild(img);
      return wrapper;
    }

    function renderFigureCard(figure) {
      const card = el("div", {
        className: "figure-card",
        role: "button",
        tabindex: "0",
        onclick: () => openFigure(figure),
        onkeydown: event => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            openFigure(figure);
          }
        }
      });

      card.appendChild(figurePreviewNode(figure));

      const body = el("div", { className: "figure-body" });
      const kicker = el("div", { className: "figure-kicker" }, [
        el("h3", { className: "figure-name", text: figure.title }),
        el("span", { className: "figure-badge", text: figure.badge || "Static" })
      ]);
      body.appendChild(kicker);
      body.appendChild(el("p", { className: "figure-description", text: figure.description || "" }));
      body.appendChild(el("div", { className: "figure-meta" }, [
        el("span", { className: "figure-chip", text: figure.category || "Figure" }),
        el("span", { className: "figure-chip", text: figure.interactiveViewId ? "Open studio" : "Preview only" })
      ]));

      if (figure.interactiveViewId) {
        const previewButton = el("button", {
          text: "Preview",
          onclick: event => {
            event.stopPropagation();
            openFigurePreview(figure);
          }
        });
        body.appendChild(el("div", { className: "detail-actions" }, [previewButton]));
      }

      card.appendChild(body);
      return card;
    }

    function renderFigureDetail(figure) {
      const panel = el("section", { className: "panel" });
      if (!figure) {
        panel.appendChild(el("div", { className: "placeholder", text: "No figure selected." }));
        return panel;
      }

      panel.appendChild(el("div", { className: "gallery-head" }, [
        el("h2", { className: "gallery-title", text: figure.title }),
        el("p", { className: "gallery-note", text: figure.description || "Figure preview." })
      ]));

      const detail = el("div", { className: "detail-panel" });
      detail.appendChild(el("div", { className: "figure-meta" }, [
        el("span", { className: "figure-chip", text: figure.category || "Figure" }),
        el("span", { className: "figure-chip", text: figure.badge || "Static" })
      ]));
      detail.appendChild(figurePreviewNode(figure));

      const actions = el("div", { className: "detail-actions" }, [
        el("button", {
          text: "Back to gallery",
          onclick: () => {
            state.activeFigureId = "";
            state.activeViewId = "gallery";
            render();
          }
        })
      ]);

      if (figure.interactiveViewId && canOpenFigureStudio(figure)) {
        actions.appendChild(el("button", {
          text: "Open studio",
          onclick: () => openFigureStudio(figure)
        }));
      }
      detail.appendChild(actions);

      const links = el("div", { className: "detail-links" }, [
        el("a", { href: figure.staticPaths?.png || figure.previewPath, target: "_blank", rel: "noopener", text: "Open PNG" }),
        el("a", { href: figure.staticPaths?.svg || figure.previewPath, target: "_blank", rel: "noopener", text: "Open SVG" }),
        el("a", { href: figure.staticPaths?.pdf || figure.previewPath, target: "_blank", rel: "noopener", text: "Open PDF" })
      ]);
      detail.appendChild(links);
      panel.appendChild(detail);
      return panel;
    }

    function renderGalleryView(view) {
      const selectedFigure = state.activeFigureId ? getFigure(state.activeFigureId) : null;
      if (selectedFigure) {
        return renderFigureDetail(selectedFigure);
      }

      const panel = el("section", { className: "panel" });
      panel.appendChild(el("div", { className: "gallery-head" }, [
        el("h2", { className: "gallery-title", text: view.title }),
        el("p", {
          className: "gallery-note",
          text: "Browse the generated static figures for this association analysis run. Open a card to jump into the matching interactive view when it is available."
        })
      ]));
      panel.appendChild(el("div", { className: "gallery-summary" }, [
        el("span", { text: `Figures: ${figures.length}` }),
        el("span", { text: `Interactive: ${figures.filter(item => item.interactiveViewId && canOpenFigureStudio(item)).length}` }),
        el("span", { text: `Preview only: ${figures.filter(item => !item.interactiveViewId).length}` })
      ]));

      const grid = el("div", { className: "figure-grid" });
      if (!figures.length) {
        grid.appendChild(el("div", { className: "placeholder", text: "No static figures were generated for this run." }));
      } else {
        for (const figure of figures) {
          grid.appendChild(renderFigureCard(figure));
        }
      }
      panel.appendChild(grid);
      return panel;
    }

    function renderAssociationLegend(dataset, colorBy) {
      const legend = el("div", { className: "legend" });
      const groupInfo = colorBy === "group2" ? dataset.groupOptions.group2 : dataset.groupOptions.group1;
      if (!groupInfo || !Array.isArray(groupInfo.order) || groupInfo.order.length === 0 || colorBy === "none") {
        legend.appendChild(el("span", { text: "No group legend available." }));
        return legend;
      }
      const colors = groupInfo.colors || {};
      for (const groupName of groupInfo.order) {
        const swatch = el("span", { className: "swatch", style: `background:${colors[groupName] || "#6b7280"}` });
        legend.appendChild(el("span", { className: "legend-item" }, [swatch, el("span", { text: groupName })]));
      }
      return legend;
    }

    function pcaMarkerNode(marker, cx, cy, size, fill, stroke) {
      const name = String(marker || "circle").toLowerCase();
      if (name === "s" || name === "square") {
        return svgEl("rect", {
          x: cx - size,
          y: cy - size,
          width: size * 2,
          height: size * 2,
          fill,
          stroke,
          "stroke-width": 1.1
        });
      }
      if (name === "^" || name === "triangle_up") {
        return svgEl("polygon", {
          points: `${cx},${cy - size * 1.25} ${cx - size * 1.1},${cy + size * 0.85} ${cx + size * 1.1},${cy + size * 0.85}`,
          fill,
          stroke,
          "stroke-width": 1.1
        });
      }
      if (name === "v" || name === "triangle_down") {
        return svgEl("polygon", {
          points: `${cx},${cy + size * 1.25} ${cx - size * 1.1},${cy - size * 0.85} ${cx + size * 1.1},${cy - size * 0.85}`,
          fill,
          stroke,
          "stroke-width": 1.1
        });
      }
      if (name === "<" || name === "triangle_left") {
        return svgEl("polygon", {
          points: `${cx - size * 1.25},${cy} ${cx + size * 0.85},${cy - size * 1.1} ${cx + size * 0.85},${cy + size * 1.1}`,
          fill,
          stroke,
          "stroke-width": 1.1
        });
      }
      if (name === ">" || name === "triangle_right") {
        return svgEl("polygon", {
          points: `${cx + size * 1.25},${cy} ${cx - size * 0.85},${cy - size * 1.1} ${cx - size * 0.85},${cy + size * 1.1}`,
          fill,
          stroke,
          "stroke-width": 1.1
        });
      }
      if (name === "d" || name === "diamond") {
        return svgEl("polygon", {
          points: `${cx},${cy - size * 1.25} ${cx - size * 1.25},${cy} ${cx},${cy + size * 1.25} ${cx + size * 1.25},${cy}`,
          fill,
          stroke,
          "stroke-width": 1.1
        });
      }
      if (name === "x") {
        const group = svgEl("g", { stroke: fill, "stroke-width": 2.0, "stroke-linecap": "round" });
        group.appendChild(svgEl("line", { x1: cx - size, y1: cy - size, x2: cx + size, y2: cy + size }));
        group.appendChild(svgEl("line", { x1: cx + size, y1: cy - size, x2: cx - size, y2: cy + size }));
        return group;
      }
      if (name === "p" || name === "pentagon") {
        const points = [];
        for (let idx = 0; idx < 5; idx++) {
          const angle = -Math.PI / 2 + idx * 2 * Math.PI / 5;
          points.push(`${cx + Math.cos(angle) * size * 1.18},${cy + Math.sin(angle) * size * 1.18}`);
        }
        return svgEl("polygon", {
          points: points.join(" "),
          fill,
          stroke,
          "stroke-width": 1.1
        });
      }
      if (name === "h" || name === "hexagon" || name === "8") {
        const count = name === "8" ? 8 : 6;
        const points = [];
        for (let idx = 0; idx < count; idx++) {
          const angle = -Math.PI / 2 + idx * 2 * Math.PI / count;
          points.push(`${cx + Math.cos(angle) * size * 1.12},${cy + Math.sin(angle) * size * 1.12}`);
        }
        return svgEl("polygon", {
          points: points.join(" "),
          fill,
          stroke,
          "stroke-width": 1.1
        });
      }
      if (name === "*" || name === "star") {
        const points = [];
        for (let idx = 0; idx < 10; idx++) {
          const angle = -Math.PI / 2 + idx * Math.PI / 5;
          const radius = idx % 2 === 0 ? size * 1.35 : size * 0.55;
          points.push(`${cx + Math.cos(angle) * radius},${cy + Math.sin(angle) * radius}`);
        }
        return svgEl("polygon", {
          points: points.join(" "),
          fill,
          stroke,
          "stroke-width": 1.1
        });
      }
      if (name === "plus") {
        const group = svgEl("g", { stroke: fill, "stroke-width": 2.0, "stroke-linecap": "round" });
        group.appendChild(svgEl("line", { x1: cx - size, y1: cy, x2: cx + size, y2: cy }));
        group.appendChild(svgEl("line", { x1: cx, y1: cy - size, x2: cx, y2: cy + size }));
        return group;
      }
      return svgEl("circle", {
        cx,
        cy,
        r: size,
        fill,
        stroke,
        "stroke-width": 1.1
      });
    }

    function pcaComponentValue(point, componentIndex, fallbackField) {
      const components = Array.isArray(point.components) ? point.components : [];
      const value = Number(components[componentIndex]);
      if (Number.isFinite(value)) return value;
      return Number(point[fallbackField] || 0);
    }

    function pcaVariancePct(dataset, componentIndex) {
      const values = dataset.varianceExplained?.components;
      if (Array.isArray(values) && Number.isFinite(Number(values[componentIndex]))) {
        return Number(values[componentIndex]);
      }
      const key = `pc${componentIndex + 1}`;
      return Number(dataset.varianceExplained?.[key]);
    }

    function pcaEnvelopePath(groupPoints) {
      if (!Array.isArray(groupPoints) || groupPoints.length === 0) return "";
      const points = [...groupPoints].sort((a, b) => a.x === b.x ? a.y - b.y : a.x - b.x);
      if (points.length === 1) {
        const p = points[0];
        const r = 16;
        return `M ${p.x - r} ${p.y} a ${r} ${r} 0 1 0 ${r * 2} 0 a ${r} ${r} 0 1 0 ${-r * 2} 0`;
      }
      if (points.length === 2) {
        const [a, b] = points;
        return `M ${a.x} ${a.y} L ${b.x} ${b.y}`;
      }
      const cross = (o, a, b) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
      const lower = [];
      for (const point of points) {
        while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], point) <= 0) lower.pop();
        lower.push(point);
      }
      const upper = [];
      for (let idx = points.length - 1; idx >= 0; idx--) {
        const point = points[idx];
        while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], point) <= 0) upper.pop();
        upper.push(point);
      }
      const hull = lower.slice(0, -1).concat(upper.slice(0, -1));
      if (hull.length < 3) return "";
      const centroid = hull.reduce((acc, p) => ({ x: acc.x + p.x, y: acc.y + p.y }), { x: 0, y: 0 });
      centroid.x /= hull.length;
      centroid.y /= hull.length;
      const padded = hull.map(p => {
        const dx = p.x - centroid.x;
        const dy = p.y - centroid.y;
        const length = Math.sqrt(dx * dx + dy * dy) || 1;
        return { x: p.x + (dx / length) * 12, y: p.y + (dy / length) * 12 };
      });
      return `M ${padded.map(p => `${p.x} ${p.y}`).join(" L ")} Z`;
    }

    function renderPcaChart(dataset, controls) {
      const width = clamp(Number(controls.width || 900), 640, 2000);
      const height = clamp(Number(controls.height || 620), 480, 1800);
      const pointSize = clamp(Number(controls.pointSize || 5), 2, 14);
      const showLabels = Boolean(controls.showLabels);
      const showGroupEnvelope = Boolean(controls.showGroupEnvelope);
      const colorBy = controls.colorBy === "group2" ? "group2" : "group1";
      const points = Array.isArray(dataset.points) ? dataset.points : [];
      const componentCount = Math.max(2, Number(dataset.componentCount || 2));
      let xComponent = clamp(Number(controls.xComponent || 1), 1, componentCount);
      let yComponent = clamp(Number(controls.yComponent || 2), 1, componentCount);
      if (xComponent === yComponent) yComponent = xComponent === 1 ? Math.min(2, componentCount) : 1;
      const xComponentIndex = xComponent - 1;
      const yComponentIndex = yComponent - 1;
      const title = dataset.title || "PCA";
      const margin = { top: 48, right: 38, bottom: 60, left: 72 };
      const innerWidth = Math.max(1, width - margin.left - margin.right);
      const innerHeight = Math.max(1, height - margin.top - margin.bottom);

      const xs = points.map(p => pcaComponentValue(p, xComponentIndex, "x"));
      const ys = points.map(p => pcaComponentValue(p, yComponentIndex, "y"));
      const xmin = Math.min(0, ...xs);
      const xmax = Math.max(0, ...xs);
      const ymin = Math.min(0, ...ys);
      const ymax = Math.max(0, ...ys);
      const xpad = Math.max(0.12 * Math.max(1e-6, xmax - xmin), 0.25);
      const ypad = Math.max(0.12 * Math.max(1e-6, ymax - ymin), 0.25);
      const x0 = xmin - xpad;
      const x1 = xmax + xpad;
      const y0 = ymin - ypad;
      const y1 = ymax + ypad;
      const sx = value => margin.left + ((value - x0) / Math.max(1e-6, x1 - x0)) * innerWidth;
      const sy = value => margin.top + ((y1 - value) / Math.max(1e-6, y1 - y0)) * innerHeight;

      const svg = svgEl("svg", {
        width,
        height,
        viewBox: `0 0 ${width} ${height}`,
        role: "img",
        "aria-label": title
      });
      svg.appendChild(svgEl("rect", { x: 0, y: 0, width, height, fill: "#ffffff" }));
      svg.appendChild(svgEl("line", {
        x1: margin.left,
        y1: sy(0),
        x2: width - margin.right,
        y2: sy(0),
        stroke: "#cbd5e1",
        "stroke-dasharray": "6 4"
      }));
      svg.appendChild(svgEl("line", {
        x1: sx(0),
        y1: margin.top,
        x2: sx(0),
        y2: height - margin.bottom,
        stroke: "#cbd5e1",
        "stroke-dasharray": "6 4"
      }));

      const axisColor = "#334155";
      svg.appendChild(svgEl("text", {
        x: width / 2,
        y: 24,
        "text-anchor": "middle",
        "font-size": 18,
        "font-weight": 700,
        fill: "#111827"
      }));
      svg.lastChild.textContent = `${title}`;

      svg.appendChild(svgEl("text", {
        x: width / 2,
        y: height - 16,
        "text-anchor": "middle",
        "font-size": 13,
        fill: axisColor
      }));
      svg.lastChild.textContent = `PC${xComponent} (${fmtPct(pcaVariancePct(dataset, xComponentIndex))})`;

      svg.appendChild(svgEl("text", {
        x: 20,
        y: height / 2,
        transform: `rotate(-90 20 ${height / 2})`,
        "text-anchor": "middle",
        "font-size": 13,
        fill: axisColor
      }));
      svg.lastChild.textContent = `PC${yComponent} (${fmtPct(pcaVariancePct(dataset, yComponentIndex))})`;

      if (showGroupEnvelope) {
        const envelopeGroups = new Map();
        for (const point of points) {
          const groupName = colorBy === "group2" ? (point.group2 || "Missing") : (point.group1 || "Missing");
          const color = colorBy === "group2" ? (point.group2Color || "#6b7280") : (point.group1Color || "#6b7280");
          const cx = sx(pcaComponentValue(point, xComponentIndex, "x"));
          const cy = sy(pcaComponentValue(point, yComponentIndex, "y"));
          if (!envelopeGroups.has(groupName)) envelopeGroups.set(groupName, { color, points: [] });
          envelopeGroups.get(groupName).points.push({ x: cx, y: cy });
        }
        for (const group of envelopeGroups.values()) {
          const pathData = pcaEnvelopePath(group.points);
          if (!pathData) continue;
          const attrs = {
            d: pathData,
            fill: group.points.length >= 3 ? group.color : "none",
            stroke: group.color,
            "stroke-width": group.points.length >= 3 ? 1.4 : 8,
            opacity: group.points.length >= 3 ? 0.18 : 0.16,
            "stroke-linejoin": "round",
            "stroke-linecap": "round"
          };
          svg.appendChild(svgEl("path", attrs));
          if (group.points.length === 2) {
            svg.appendChild(svgEl("path", {
              d: pathData,
              fill: "none",
              stroke: group.color,
              "stroke-width": 1.3,
              opacity: 0.90,
              "stroke-linecap": "round"
            }));
          }
        }
      }

      for (const point of points) {
        const color = colorBy === "group2" ? (point.group2Color || "#6b7280") : (point.group1Color || "#6b7280");
        const marker = colorBy === "group2" ? (point.group1Marker || "circle") : "circle";
        const cx = sx(pcaComponentValue(point, xComponentIndex, "x"));
        const cy = sy(pcaComponentValue(point, yComponentIndex, "y"));
        const markerNode = pcaMarkerNode(marker, cx, cy, pointSize, color, "#ffffff");
        const markerTitle = svgEl("title");
        markerTitle.textContent = point.id || point.label || "";
        markerNode.appendChild(markerTitle);
        svg.appendChild(markerNode);

        if (showLabels) {
          const dx = cx >= width / 2 ? -8 : 8;
          const anchor = cx >= width / 2 ? "end" : "start";
          const label = svgEl("text", {
            x: cx + dx,
            y: cy - 6,
            "text-anchor": anchor,
            "font-size": 10,
            fill: "#334155"
          });
          label.textContent = point.label || point.id;
          svg.appendChild(label);
        }
      }
      return svg;
    }

    function computeLinearFit(points) {
      const clean = points.filter(p => Number.isFinite(p.x) && Number.isFinite(p.y));
      if (clean.length < 2) return null;
      const xs = clean.map(p => p.x);
      const ys = clean.map(p => p.y);
      const n = clean.length;
      const xMean = xs.reduce((a, b) => a + b, 0) / n;
      const yMean = ys.reduce((a, b) => a + b, 0) / n;
      let sxx = 0;
      let sxy = 0;
      let syy = 0;
      for (let i = 0; i < n; i++) {
        const dx = xs[i] - xMean;
        const dy = ys[i] - yMean;
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
      }
      if (sxx <= 0 || syy <= 0) return null;
      const slope = sxy / sxx;
      const intercept = yMean - slope * xMean;
      const pearson = sxy / Math.sqrt(sxx * syy);
      const fitted = xs.map(x => intercept + slope * x);
      const residualSs = ys.reduce((acc, y, idx) => acc + Math.pow(y - fitted[idx], 2), 0);
      const dof = n - 2;
      const residualSe = dof > 0 ? Math.sqrt(residualSs / dof) : null;

      const rankedX = xs.map((v, i) => ({ v, i })).sort((a, b) => a.v - b.v);
      const rx = new Array(n);
      for (let i = 0; i < rankedX.length; i++) rx[rankedX[i].i] = i + 1;
      const rankedY = ys.map((v, i) => ({ v, i })).sort((a, b) => a.v - b.v);
      const ry = new Array(n);
      for (let i = 0; i < rankedY.length; i++) ry[rankedY[i].i] = i + 1;
      const rxMean = rx.reduce((a, b) => a + b, 0) / n;
      const ryMean = ry.reduce((a, b) => a + b, 0) / n;
      let rsxx = 0;
      let rsyy = 0;
      let rsxy = 0;
      for (let i = 0; i < n; i++) {
        const dx = rx[i] - rxMean;
        const dy = ry[i] - ryMean;
        rsxx += dx * dx;
        rsyy += dy * dy;
        rsxy += dx * dy;
      }
      const spearman = rsxx > 0 && rsyy > 0 ? rsxy / Math.sqrt(rsxx * rsyy) : null;
      return { slope, intercept, pearson, spearman, xMean, sxx, residualSe, dof };
    }

    function approximateTCritical(dof) {
      const df = Math.max(1, Math.floor(Number(dof || 1)));
      const table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042
      };
      if (table[df]) return table[df];
      if (df <= 40) return 2.021;
      if (df <= 60) return 2.000;
      if (df <= 120) return 1.980;
      return 1.960;
    }

    function renderAssociationChart(dataset, controls) {
      const width = clamp(Number(controls.width || 900), 640, 2000);
      const height = clamp(Number(controls.height || 640), 480, 1800);
      const pointSize = clamp(Number(controls.pointSize || 5), 2, 14);
      const alpha = clamp(Number(controls.alpha || 0.85), 0.15, 1.0);
      const showLabels = Boolean(controls.showLabels);
      const showRegression = Boolean(controls.showRegression);
      const selected = findAssociationEdge(dataset, controls) || (Array.isArray(dataset.topEdges) ? dataset.topEdges[0] : null);
      const title = dataset.title || "Association Scatter";
      const margin = { top: 48, right: 38, bottom: 60, left: 72 };
      const innerWidth = Math.max(1, width - margin.left - margin.right);
      const innerHeight = Math.max(1, height - margin.top - margin.bottom);
      const samples = dataset.sampleIds || [];

      if (!selected) {
        return el("div", { className: "placeholder", text: "No regression payload available for the selected type." });
      }

      const xs = selected.x || [];
      const ys = selected.y || [];
      const finiteX = xs.filter(v => Number.isFinite(v));
      const finiteY = ys.filter(v => Number.isFinite(v));
      const xmin = Math.min(0, ...finiteX);
      const xmax = Math.max(0, ...finiteX);
      const ymin = Math.min(0, ...finiteY);
      const ymax = Math.max(0, ...finiteY);
      const xpad = Math.max(0.12 * Math.max(1e-6, xmax - xmin), 0.25);
      const ypad = Math.max(0.12 * Math.max(1e-6, ymax - ymin), 0.25);
      const x0 = xmin - xpad;
      const x1 = xmax + xpad;
      const y0 = ymin - ypad;
      const y1 = ymax + ypad;
      const sx = value => margin.left + ((value - x0) / Math.max(1e-6, x1 - x0)) * innerWidth;
      const sy = value => margin.top + ((y1 - value) / Math.max(1e-6, y1 - y0)) * innerHeight;
      const moduleColor = selected.pointColor || selected.moduleColor || "#4c78a8";

      const points = samples.map((sampleId, idx) => {
        const x = xs[idx] === null || xs[idx] === undefined ? Number.NaN : Number(xs[idx]);
        const y = ys[idx] === null || ys[idx] === undefined ? Number.NaN : Number(ys[idx]);
        return {
          id: sampleId,
          x,
          y
        };
      }).filter(point => Number.isFinite(point.x) && Number.isFinite(point.y));

      const stats = computeLinearFit(points);
      const svg = svgEl("svg", {
        width,
        height,
        viewBox: `0 0 ${width} ${height}`,
        role: "img",
        "aria-label": title
      });

      svg.appendChild(svgEl("rect", { x: 0, y: 0, width, height, fill: "#ffffff" }));
      svg.appendChild(svgEl("line", {
        x1: margin.left,
        y1: sy(0),
        x2: width - margin.right,
        y2: sy(0),
        stroke: "#cbd5e1",
        "stroke-dasharray": "6 4"
      }));
      svg.appendChild(svgEl("line", {
        x1: sx(0),
        y1: margin.top,
        x2: sx(0),
        y2: height - margin.bottom,
        stroke: "#cbd5e1",
        "stroke-dasharray": "6 4"
      }));

      svg.appendChild(svgEl("text", {
        x: width / 2,
        y: 24,
        "text-anchor": "middle",
        "font-size": 18,
        "font-weight": 700,
        fill: "#111827"
      }));
      svg.lastChild.textContent = `${title}: ${selected.label || `${selected.gene} vs ${selected.metabolite}`}`;

      svg.appendChild(svgEl("text", {
        x: width / 2,
        y: height - 16,
        "text-anchor": "middle",
        "font-size": 13,
        fill: "#334155"
      }));
      svg.lastChild.textContent = selected.xLabel || selected.gene;

      svg.appendChild(svgEl("text", {
        x: 20,
        y: height / 2,
        transform: `rotate(-90 20 ${height / 2})`,
        "text-anchor": "middle",
        "font-size": 13,
        fill: "#334155"
      }));
      svg.lastChild.textContent = selected.yLabel || selected.metabolite;

      if (showRegression && stats) {
        const xStart = x0;
        const xEnd = x1;
        const yStart = stats.intercept + stats.slope * xStart;
        const yEnd = stats.intercept + stats.slope * xEnd;
        if (stats.residualSe !== null && stats.dof > 0 && stats.sxx > 0) {
          const tValue = approximateTCritical(stats.dof);
          const upper = [];
          const lower = [];
          for (let idx = 0; idx < 80; idx++) {
            const xValue = xStart + (xEnd - xStart) * idx / 79;
            const yFit = stats.intercept + stats.slope * xValue;
            const seMean = stats.residualSe * Math.sqrt((1 / points.length) + Math.pow(xValue - stats.xMean, 2) / stats.sxx);
            const delta = tValue * seMean;
            if (!Number.isFinite(delta)) continue;
            upper.push({ x: sx(xValue), y: sy(yFit + delta) });
            lower.push({ x: sx(xValue), y: sy(yFit - delta) });
          }
          if (upper.length > 1 && lower.length > 1) {
            const bandPoints = upper.concat(lower.reverse()).map(p => `${p.x},${p.y}`).join(" ");
            svg.appendChild(svgEl("polygon", {
              points: bandPoints,
              fill: moduleColor,
              opacity: 0.16,
              stroke: "none"
            }));
          }
        }
        svg.appendChild(svgEl("line", {
          x1: sx(xStart),
          y1: sy(yStart),
          x2: sx(xEnd),
          y2: sy(yEnd),
          stroke: "#111827",
          "stroke-width": 1.6
        }));
      }

      const rLabel = selected.rLabel || "r";
      const realtimeR = rLabel === "rho" ? (stats ? stats.spearman : null) : (stats ? stats.pearson : null);
      const rValue = realtimeR !== null && realtimeR !== undefined ? Number(realtimeR) : (
        selected.rValue !== null && selected.rValue !== undefined ? Number(selected.rValue) : null
      );
      const rText = Number.isFinite(rValue) ? `${rLabel} = ${rValue.toFixed(2)}` : `${rLabel} = NA`;
      const rGroup = svgEl("g");
      rGroup.appendChild(svgEl("rect", {
        x: margin.left + 10,
        y: margin.top + 10,
        width: Math.max(58, rText.length * 8 + 14),
        height: 22,
        fill: "#ffffff",
        opacity: 0.75,
        stroke: "none",
        rx: 3
      }));
      const rTextNode = svgEl("text", {
        x: margin.left + 17,
        y: margin.top + 26,
        "font-size": 12,
        "font-weight": 700,
        fill: "#111827"
      });
      rTextNode.textContent = rText;
      rGroup.appendChild(rTextNode);
      svg.appendChild(rGroup);

      for (const point of points) {
        const cx = sx(point.x);
        const cy = sy(point.y);
        const circle = svgEl("circle", {
          cx,
          cy,
          r: pointSize,
          fill: moduleColor,
          opacity: alpha,
          stroke: "#ffffff",
          "stroke-width": 1.0
        });
        circle.dataset.sampleId = point.id;
        svg.appendChild(circle);

        if (showLabels) {
          const dx = cx >= width / 2 ? -8 : 8;
          const anchor = cx >= width / 2 ? "end" : "start";
          const label = svgEl("text", {
            x: cx + dx,
            y: cy - 6,
            "text-anchor": anchor,
            "font-size": 10,
            fill: "#334155"
          });
          label.textContent = point.id;
          svg.appendChild(label);
        }
      }

      const summary = el("div", { className: "legend" });
      const chips = [`Module: ${selected.module || "Unassigned"}`, `Samples: ${points.length}`];
      if (selected.edgeWeight !== undefined && selected.edgeWeight !== null) chips.splice(1, 0, `EdgeWeight: ${Number(selected.edgeWeight).toFixed(3)}`);
      for (const text of chips) summary.appendChild(el("span", { className: "legend-item", text }));

      return { svg, summary, selected };
    }

    function formatSignificanceMetric(metric) {
      return metric === "FDR" ? "FDR" : "PValue";
    }

    function sortHeatmapItems(items, mode) {
      const sorted = [...items];
      if (mode === "max_abs_rho") {
        sorted.sort((a, b) => {
          const delta = Number(b.maxAbsRho || 0) - Number(a.maxAbsRho || 0);
          return delta || String(a.label || a.id).localeCompare(String(b.label || b.id));
        });
      } else if (mode === "significance") {
        sorted.sort((a, b) => {
          const av = a.minSignificance !== null && a.minSignificance !== undefined && Number.isFinite(Number(a.minSignificance)) ? Number(a.minSignificance) : Number.POSITIVE_INFINITY;
          const bv = b.minSignificance !== null && b.minSignificance !== undefined && Number.isFinite(Number(b.minSignificance)) ? Number(b.minSignificance) : Number.POSITIVE_INFINITY;
          return (av - bv) || String(a.label || a.id).localeCompare(String(b.label || b.id));
        });
      } else if (mode === "name") {
        sorted.sort((a, b) => String(a.label || a.id).localeCompare(String(b.label || b.id)));
      } else {
        sorted.sort((a, b) => Number(a.defaultRank || 0) - Number(b.defaultRank || 0));
      }
      return sorted;
    }

    function hexToRgb(hex) {
      const clean = String(hex || "").replace("#", "");
      const value = clean.length === 3
        ? clean.split("").map(ch => ch + ch).join("")
        : clean.padEnd(6, "0").slice(0, 6);
      return {
        r: parseInt(value.slice(0, 2), 16),
        g: parseInt(value.slice(2, 4), 16),
        b: parseInt(value.slice(4, 6), 16)
      };
    }

    function rgbToHex(rgb) {
      const toHex = value => clamp(Math.round(value), 0, 255).toString(16).padStart(2, "0");
      return `#${toHex(rgb.r)}${toHex(rgb.g)}${toHex(rgb.b)}`;
    }

    function mixHex(a, b, t) {
      const ac = hexToRgb(a);
      const bc = hexToRgb(b);
      return rgbToHex({
        r: ac.r + (bc.r - ac.r) * t,
        g: ac.g + (bc.g - ac.g) * t,
        b: ac.b + (bc.b - ac.b) * t
      });
    }

    function heatmapColor(value, extent, paletteName) {
      if (!Number.isFinite(value)) return "#f8fafc";
      const palettes = {
        rdbu: ["#2166ac", "#f7f7f7", "#b2182b"],
        blueorange: ["#2563eb", "#f8fafc", "#ea580c"],
        purplegreen: ["#7e22ce", "#f7f7f7", "#15803d"]
      };
      const palette = palettes[paletteName] || palettes.rdbu;
      const maxAbs = Math.max(
        Math.abs(Number(extent?.min ?? -1)),
        Math.abs(Number(extent?.max ?? 1)),
        0.25
      );
      const normalized = clamp(value / maxAbs, -1, 1);
      if (normalized < 0) return mixHex(palette[0], palette[1], normalized + 1);
      return mixHex(palette[1], palette[2], normalized);
    }

    function contrastTextColor(fill) {
      const rgb = hexToRgb(fill);
      const luminance = (0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b) / 255;
      return luminance < 0.54 ? "#ffffff" : "#111827";
    }

    function prepareHeatmapGrid(dataset, controls) {
      const modules = sortHeatmapItems(dataset.modules || [], controls.rowSort || "default")
        .slice(0, clamp(Math.round(Number(controls.topModules || 10)), 1, Math.max(1, (dataset.modules || []).length)));
      const metabolites = sortHeatmapItems(dataset.metabolites || [], controls.columnSort || "significance")
        .slice(0, clamp(Math.round(Number(controls.topMetabolites || 20)), 1, Math.max(1, (dataset.metabolites || []).length)));
      const moduleIds = new Set(modules.map(item => item.id));
      const metaboliteIds = new Set(metabolites.map(item => item.id));
      const cellMap = new Map();
      for (const cell of dataset.cells || []) {
        if (moduleIds.has(cell.module) && metaboliteIds.has(cell.metabolite)) {
          cellMap.set(`${cell.module}||${cell.metabolite}`, cell);
        }
      }
      return { modules, metabolites, cellMap };
    }

    function renderModuleHeatmapChart(dataset, controls) {
      const width = clamp(Number(controls.width || 980), 720, 2400);
      const height = clamp(Number(controls.height || 720), 520, 2000);
      const showValues = Boolean(controls.showValues);
      const showStars = Boolean(controls.showStars);
      const palette = controls.palette || "rdbu";
      const grid = prepareHeatmapGrid(dataset, controls);
      const modules = grid.modules;
      const metabolites = grid.metabolites;
      if (!modules.length || !metabolites.length) return null;

      const rowLabelWidth = Math.min(220, Math.max(92, 8 * Math.max(...modules.map(item => String(item.label || item.id).length)) + 18));
      const colLabelHeight = Math.min(190, Math.max(92, 7 * Math.max(...metabolites.map(item => String(item.label || item.id).length)) + 20));
      const margin = { top: 58, right: 130, bottom: colLabelHeight + 48, left: rowLabelWidth + 22 };
      const innerWidth = Math.max(1, width - margin.left - margin.right);
      const innerHeight = Math.max(1, height - margin.top - margin.bottom);
      const cellSize = Math.max(8, Math.min(innerWidth / metabolites.length, innerHeight / modules.length));
      const gridWidth = cellSize * metabolites.length;
      const gridHeight = cellSize * modules.length;
      const x0 = margin.left;
      const y0 = margin.top;
      const title = dataset.title || "Module-Metabolite Association Heatmap";
      const metricLabel = formatSignificanceMetric(dataset.significanceMetric);

      const svg = svgEl("svg", {
        width,
        height,
        viewBox: `0 0 ${width} ${height}`,
        role: "img",
        "aria-label": title
      });
      svg.appendChild(svgEl("rect", { x: 0, y: 0, width, height, fill: "#ffffff" }));

      const titleText = svgEl("text", {
        x: width / 2,
        y: 24,
        "text-anchor": "middle",
        "font-size": 18,
        "font-weight": 700,
        fill: "#111827"
      });
      titleText.textContent = title;
      svg.appendChild(titleText);

      for (let rowIndex = 0; rowIndex < modules.length; rowIndex++) {
        const module = modules[rowIndex];
        const y = y0 + rowIndex * cellSize;
        const label = svgEl("text", {
          x: x0 - 8,
          y: y + cellSize / 2 + 4,
          "text-anchor": "end",
          "font-size": 11,
          fill: "#334155"
        });
        label.textContent = module.label || module.id;
        svg.appendChild(label);
      }

      for (let colIndex = 0; colIndex < metabolites.length; colIndex++) {
        const metabolite = metabolites[colIndex];
        const x = x0 + colIndex * cellSize;
        const label = svgEl("text", {
          x: x + cellSize / 2,
          y: y0 + gridHeight + 10,
          transform: `rotate(55 ${x + cellSize / 2} ${y0 + gridHeight + 10})`,
          "text-anchor": "start",
          "font-size": 10,
          fill: "#334155"
        });
        label.textContent = metabolite.label || metabolite.id;
        svg.appendChild(label);
      }

      for (let rowIndex = 0; rowIndex < modules.length; rowIndex++) {
        for (let colIndex = 0; colIndex < metabolites.length; colIndex++) {
          const module = modules[rowIndex];
          const metabolite = metabolites[colIndex];
          const cell = grid.cellMap.get(`${module.id}||${metabolite.id}`);
          const rho = cell ? Number(cell.rho) : NaN;
          const fill = heatmapColor(rho, dataset.rhoExtent, palette);
          const x = x0 + colIndex * cellSize;
          const y = y0 + rowIndex * cellSize;
          const rect = svgEl("rect", {
            x,
            y,
            width: cellSize,
            height: cellSize,
            fill,
            stroke: "#f1f5f9",
            "stroke-width": 0.8
          });
          if (cell) {
            rect.dataset.module = module.id;
            rect.dataset.metabolite = metabolite.id;
            rect.dataset.rho = String(cell.rho);
          }
          svg.appendChild(rect);

          if (cell && (showValues || showStars) && cellSize >= 16) {
            const textParts = [];
            if (showValues) textParts.push(Number(rho).toFixed(2));
            if (showStars && cell.star) textParts.push(cell.star);
            if (textParts.length) {
              const valueText = svgEl("text", {
                x: x + cellSize / 2,
                y: y + cellSize / 2 + 4,
                "text-anchor": "middle",
                "font-size": cellSize < 24 ? 9 : 10,
                "font-weight": cell.star ? 700 : 500,
                fill: contrastTextColor(fill)
              });
              valueText.textContent = textParts.join(" ");
              svg.appendChild(valueText);
            }
          }
        }
      }

      svg.appendChild(svgEl("rect", {
        x: x0,
        y: y0,
        width: gridWidth,
        height: gridHeight,
        fill: "none",
        stroke: "#94a3b8",
        "stroke-width": 1
      }));

      const legendX = x0 + gridWidth + 34;
      const legendY = y0 + 10;
      const legendHeight = Math.min(220, Math.max(140, gridHeight - 20));
      const legendWidth = 14;
      const stops = 80;
      for (let i = 0; i < stops; i++) {
        const t0 = i / stops;
        const rho = Number(dataset.rhoExtent?.max || 1) - t0 * (Number(dataset.rhoExtent?.max || 1) - Number(dataset.rhoExtent?.min || -1));
        svg.appendChild(svgEl("rect", {
          x: legendX,
          y: legendY + t0 * legendHeight,
          width: legendWidth,
          height: legendHeight / stops + 0.8,
          fill: heatmapColor(rho, dataset.rhoExtent, palette)
        }));
      }
      svg.appendChild(svgEl("rect", {
        x: legendX,
        y: legendY,
        width: legendWidth,
        height: legendHeight,
        fill: "none",
        stroke: "#94a3b8",
        "stroke-width": 1
      }));

      const maxLabel = svgEl("text", { x: legendX + 22, y: legendY + 4, "font-size": 11, fill: "#334155" });
      maxLabel.textContent = Number(dataset.rhoExtent?.max || 1).toFixed(2);
      svg.appendChild(maxLabel);
      const zeroLabel = svgEl("text", { x: legendX + 22, y: legendY + legendHeight / 2 + 4, "font-size": 11, fill: "#334155" });
      zeroLabel.textContent = "0";
      svg.appendChild(zeroLabel);
      const minLabel = svgEl("text", { x: legendX + 22, y: legendY + legendHeight + 4, "font-size": 11, fill: "#334155" });
      minLabel.textContent = Number(dataset.rhoExtent?.min || -1).toFixed(2);
      svg.appendChild(minLabel);
      const legendLabel = svgEl("text", {
        x: legendX - 8,
        y: legendY + legendHeight / 2,
        transform: `rotate(-90 ${legendX - 8} ${legendY + legendHeight / 2})`,
        "text-anchor": "middle",
        "font-size": 12,
        fill: "#334155"
      });
      legendLabel.textContent = "Spearman rho";
      svg.appendChild(legendLabel);

      const axisLabel = svgEl("text", {
        x: x0 + gridWidth / 2,
        y: height - 16,
        "text-anchor": "middle",
        "font-size": 12,
        fill: "#334155"
      });
      axisLabel.textContent = "Metabolite";
      svg.appendChild(axisLabel);

      const rowAxisLabel = svgEl("text", {
        x: 18,
        y: y0 + gridHeight / 2,
        transform: `rotate(-90 18 ${y0 + gridHeight / 2})`,
        "text-anchor": "middle",
        "font-size": 12,
        fill: "#334155"
      });
      rowAxisLabel.textContent = "Module";
      svg.appendChild(rowAxisLabel);

      const subtitle = svgEl("text", {
        x: width / 2,
        y: 42,
        "text-anchor": "middle",
        "font-size": 11,
        fill: "#64748b"
      });
      subtitle.textContent = `Stars: ${metricLabel} <= 0.05/0.01/0.001`;
      svg.appendChild(subtitle);

      return { svg, modules, metabolites };
    }

    function renderModuleHeatmapSummary(dataset, rendered) {
      const legend = el("div", { className: "legend" });
      legend.appendChild(el("span", { className: "legend-item", text: `Modules shown: ${rendered.modules.length}/${(dataset.modules || []).length}` }));
      legend.appendChild(el("span", { className: "legend-item", text: `Metabolites shown: ${rendered.metabolites.length}/${(dataset.metabolites || []).length}` }));
      legend.appendChild(el("span", { className: "legend-item", text: `Cells: ${(dataset.cells || []).length}` }));
      legend.appendChild(el("span", { className: "legend-item", text: `Significance: ${formatSignificanceMetric(dataset.significanceMetric)}` }));
      return legend;
    }

    function normalizeSearch(value) {
      return String(value || "").trim().toLowerCase();
    }

    function filterNetworkEdges(dataset, controls) {
      return dataset.edges || [];
    }

    function prepareNetworkGraph(dataset, controls) {
      const edges = filterNetworkEdges(dataset, controls);
      const nodeLookup = new Map((dataset.nodes || []).map(node => [node.id, node]));
      const activeNodeIds = new Set();
      for (const edge of edges) {
        activeNodeIds.add(edge.source);
        activeNodeIds.add(edge.target);
      }
      const genes = (dataset.nodes || [])
        .filter(node => node.type === "gene" && activeNodeIds.has(node.id))
        .sort((a, b) => {
          const aGrey = String(a.module || "grey").toLowerCase() === "grey" ? 1 : 0;
          const bGrey = String(b.module || "grey").toLowerCase() === "grey" ? 1 : 0;
          return aGrey - bGrey
            || Number(b.moduleSize || 0) - Number(a.moduleSize || 0)
            || String(a.module || "").localeCompare(String(b.module || ""))
            || Number(b.degree || 0) - Number(a.degree || 0)
            || Number(b.maxAbsWeight || 0) - Number(a.maxAbsWeight || 0)
            || String(a.label).localeCompare(String(b.label));
        });
      const metabolites = (dataset.nodes || [])
        .filter(node => node.type === "metabolite" && activeNodeIds.has(node.id))
        .sort((a, b) => Number(b.degree || 0) - Number(a.degree || 0) || String(a.label).localeCompare(String(b.label)));
      const adjacency = new Map();
      for (const edge of edges) {
        if (!adjacency.has(edge.source)) adjacency.set(edge.source, new Set());
        if (!adjacency.has(edge.target)) adjacency.set(edge.target, new Set());
        adjacency.get(edge.source).add(edge.target);
        adjacency.get(edge.target).add(edge.source);
      }
      return { edges, nodes: [...genes, ...metabolites], genes, metabolites, nodeLookup, adjacency };
    }

    function nodeDetailText(node) {
      if (!node) return "";
      const typeLabel = node.type === "gene" ? "Gene" : "Metabolite";
      const lines = [
        `${typeLabel}: ${node.label}`,
        `Degree: ${node.degree}`,
        `Max |EdgeWeight|: ${Number(node.maxAbsWeight || 0).toFixed(3)}`,
        `Positive edges: ${node.positiveEdges || 0}`,
        `Negative edges: ${node.negativeEdges || 0}`
      ];
      if (node.type === "gene") {
        lines.push(`Module: ${node.module || "grey"}`);
        if (node.kME !== null && node.kME !== undefined) lines.push(`kME: ${Number(node.kME).toFixed(3)}`);
      }
      return lines.join("\\n");
    }

    function edgeColor(edge) {
      return edge.sign === "negative" ? "#2563eb" : "#dc2626";
    }

    function nodeColor(node) {
      if (!node) return "#9ca3af";
      if (node.type === "gene") return node.color || node.moduleColor || "#9ca3af";
      return node.color || node.moduleColor || "#c9ad85";
    }

    function edgeDetailText(edge) {
      return [
        `${edge.gene} - ${edge.metabolite}`,
        `EdgeWeight: ${Number(edge.edgeWeight || 0).toFixed(3)}`,
        `Sign: ${edge.sign}`,
        `ModelSupportCount: ${edge.modelSupportCount}`,
        `ScreenSupportCount: ${edge.screenSupportCount}`,
        edge.rraRank !== null && edge.rraRank !== undefined ? `RRARank: ${edge.rraRank}` : "",
        edge.spearmanRho !== null && edge.spearmanRho !== undefined ? `Spearman rho: ${Number(edge.spearmanRho).toFixed(3)}` : ""
      ].filter(Boolean).join("\\n");
    }

    function polarPoint(cx, cy, radius, theta) {
      return { x: cx + radius * Math.cos(theta), y: cy + radius * Math.sin(theta) };
    }

    function computeCircosLayout(genes, metabolites) {
      const nGene = genes.length;
      const nMetabolite = metabolites.length;
      const nTotal = nGene + nMetabolite;
      if (!nTotal) return new Map();

      const fullCircle = Math.PI * 2;
      const meanItemSpan = fullCircle / nTotal;
      let itemGap = Math.min(Math.PI / 400, meanItemSpan * 0.10);
      let groupGap = Math.max(7 * Math.PI / 180, itemGap * 8);
      let totalGap = Math.max(0, nTotal - 2) * itemGap + 2 * groupGap;
      if (totalGap >= fullCircle * 0.92) {
        itemGap = meanItemSpan * 0.04;
        groupGap = Math.max(4 * Math.PI / 180, itemGap * 6);
        totalGap = Math.max(0, nTotal - 2) * itemGap + 2 * groupGap;
      }

      let itemWidth = (fullCircle - totalGap) / nTotal;
      if (itemWidth <= 0) {
        itemGap = 0;
        groupGap = 0;
        itemWidth = fullCircle / nTotal;
      }

      const layout = new Map();
      let currentAngle = Math.PI * 0.76 + groupGap / 2;
      const assign = (nodes, nodeType, afterGroupGap) => {
        nodes.forEach((node, index) => {
          const thetaStart = currentAngle;
          const thetaEnd = thetaStart + itemWidth;
          layout.set(node.id, {
            thetaStart,
            thetaEnd,
            thetaMid: (thetaStart + thetaEnd) / 2,
            nodeType
          });
          currentAngle = thetaEnd + (index < nodes.length - 1 ? itemGap : afterGroupGap);
        });
      };
      assign(genes, "gene", groupGap);
      assign(metabolites, "metabolite", groupGap);
      return layout;
    }

    function annularPath(cx, cy, innerRadius, outerRadius, thetaStart, thetaEnd) {
      const largeArc = Math.abs(thetaEnd - thetaStart) > Math.PI ? 1 : 0;
      const p1 = polarPoint(cx, cy, outerRadius, thetaStart);
      const p2 = polarPoint(cx, cy, outerRadius, thetaEnd);
      const p3 = polarPoint(cx, cy, innerRadius, thetaEnd);
      const p4 = polarPoint(cx, cy, innerRadius, thetaStart);
      return [
        `M ${p1.x.toFixed(3)} ${p1.y.toFixed(3)}`,
        `A ${outerRadius.toFixed(3)} ${outerRadius.toFixed(3)} 0 ${largeArc} 1 ${p2.x.toFixed(3)} ${p2.y.toFixed(3)}`,
        `L ${p3.x.toFixed(3)} ${p3.y.toFixed(3)}`,
        `A ${innerRadius.toFixed(3)} ${innerRadius.toFixed(3)} 0 ${largeArc} 0 ${p4.x.toFixed(3)} ${p4.y.toFixed(3)}`,
        "Z"
      ].join(" ");
    }

    function chordPath(cx, cy, thetaStart, thetaEnd, radius, tension = 0.18) {
      const p1 = polarPoint(cx, cy, radius, thetaStart);
      const p4 = polarPoint(cx, cy, radius, thetaEnd);
      const p2 = polarPoint(cx, cy, radius * tension, thetaStart);
      const p3 = polarPoint(cx, cy, radius * tension, thetaEnd);
      return `M ${p1.x.toFixed(3)} ${p1.y.toFixed(3)} C ${p2.x.toFixed(3)} ${p2.y.toFixed(3)}, ${p3.x.toFixed(3)} ${p3.y.toFixed(3)}, ${p4.x.toFixed(3)} ${p4.y.toFixed(3)}`;
    }

    function pointChordPath(source, target, cx, cy, tension = 0.20) {
      const p2 = { x: cx + (source.x - cx) * tension, y: cy + (source.y - cy) * tension };
      const p3 = { x: cx + (target.x - cx) * tension, y: cy + (target.y - cy) * tension };
      return `M ${source.x.toFixed(3)} ${source.y.toFixed(3)} C ${p2.x.toFixed(3)} ${p2.y.toFixed(3)}, ${p3.x.toFixed(3)} ${p3.y.toFixed(3)}, ${target.x.toFixed(3)} ${target.y.toFixed(3)}`;
    }

    function networkSelectionState(graph, positions, controls) {
      const selectedNodeId = String(controls.selectedNodeId || "");
      const neighborIds = selectedNodeId && graph.adjacency.has(selectedNodeId) ? graph.adjacency.get(selectedNodeId) : new Set();
      const hasSelection = Boolean(selectedNodeId && graph.nodeLookup.has(selectedNodeId));
      const selectedVisible = selectedNodeId && positions.has(selectedNodeId);
      return { selectedNodeId, neighborIds, hasSelection, selectedVisible };
    }

    function biasColor(node) {
      const bias = Number.isFinite(Number(node.directionBias)) ? Number(node.directionBias) : (() => {
        const total = Number(node.positiveEdges || 0) + Number(node.negativeEdges || 0);
        return total ? (Number(node.positiveEdges || 0) - Number(node.negativeEdges || 0)) / total : 0;
      })();
      return bias >= 0 ? mixHex("#f8fafc", "#dc2626", Math.min(1, Math.abs(bias))) : mixHex("#f8fafc", "#2563eb", Math.min(1, Math.abs(bias)));
    }

    function signedHeatColor(value, scale) {
      const limit = Math.max(1e-6, Number(scale || 1));
      const normalized = clamp((Number(value || 0) + limit) / (2 * limit), 0, 1);
      return heatmapColor(normalized * 2 - 1, { min: -1, max: 1 }, "rdbu");
    }

    function addNetworkTitle(svg, dataset, graph, layoutName, width) {
      const title = svgEl("text", {
        x: width / 2,
        y: 28,
        "text-anchor": "middle",
        "font-size": 18,
        "font-weight": 700,
        fill: "#111827"
      });
      title.textContent = dataset.title || "Network Explorer";
      svg.appendChild(title);

      const subtitle = svgEl("text", {
        x: width / 2,
        y: 48,
        "text-anchor": "middle",
        "font-size": 11,
        fill: "#64748b"
      });
      subtitle.textContent = `${layoutName}; ${graph.edges.length} edges, ${graph.genes.length} genes, ${graph.metabolites.length} metabolites`;
      svg.appendChild(subtitle);
    }

    function addNetworkLegend(svg, x, y, mode) {
      const legend = svgEl("g", {});
      legend.appendChild(svgEl("rect", { x: x - 16, y: y - 22, width: 172, height: mode === "cnet" ? 96 : 118, fill: "#ffffff", stroke: "#d7dde5", rx: 8 }));
      const legendTitle = svgEl("text", { x, y, "font-size": 12, "font-weight": 700, fill: "#334155" });
      legendTitle.textContent = "Legend";
      legend.appendChild(legendTitle);
      legend.appendChild(svgEl("circle", { cx: x + 8, cy: y + 24, r: 7, fill: "#9ca3af", stroke: "#ffffff", "stroke-width": 1 }));
      const geneLabel = svgEl("text", { x: x + 24, y: y + 28, "font-size": 11, fill: "#334155" });
      geneLabel.textContent = "Gene module";
      legend.appendChild(geneLabel);
      legend.appendChild(svgEl("circle", { cx: x + 8, cy: y + 48, r: 7, fill: "#c9ad85", stroke: "#ffffff", "stroke-width": 1 }));
      const metabLabel = svgEl("text", { x: x + 24, y: y + 52, "font-size": 11, fill: "#334155" });
      metabLabel.textContent = "Metabolite";
      legend.appendChild(metabLabel);
      if (mode === "cnet") {
        legend.appendChild(svgEl("line", { x1: x, y1: y + 72, x2: x + 34, y2: y + 72, stroke: "#8b5cf6", "stroke-width": 3, opacity: 0.75 }));
        const edgeLabel = svgEl("text", { x: x + 44, y: y + 76, "font-size": 11, fill: "#334155" });
        edgeLabel.textContent = "Metabolite edge";
        legend.appendChild(edgeLabel);
      } else {
        legend.appendChild(svgEl("line", { x1: x, y1: y + 72, x2: x + 34, y2: y + 72, stroke: "#dc2626", "stroke-width": 3, opacity: 0.75 }));
        const positiveLabel = svgEl("text", { x: x + 44, y: y + 76, "font-size": 11, fill: "#334155" });
        positiveLabel.textContent = "Positive";
        legend.appendChild(positiveLabel);
        legend.appendChild(svgEl("line", { x1: x, y1: y + 96, x2: x + 34, y2: y + 96, stroke: "#2563eb", "stroke-width": 3, opacity: 0.75 }));
        const negativeLabel = svgEl("text", { x: x + 44, y: y + 100, "font-size": 11, fill: "#334155" });
        negativeLabel.textContent = "Negative";
        legend.appendChild(negativeLabel);
      }
      svg.appendChild(legend);
    }

    function addTrackAnnotationLegend(svg, x, y) {
      const rows = [
        ["track 1", "sector strip"],
        ["track 2", "group-wise mean"],
        ["track 3", "mean z-score heatmap"],
        ["track 4", "weighted degree"],
        ["track 5", "module/core strength"],
        ["track 6", "direction bias"]
      ];
      const legend = svgEl("g", {});
      legend.appendChild(svgEl("rect", { x: x - 16, y: y - 22, width: 210, height: 178, fill: "#ffffff", stroke: "#d7dde5", rx: 8 }));
      const title = svgEl("text", { x, y, "font-size": 12, "font-weight": 700, fill: "#334155" });
      title.textContent = "Track annotations";
      legend.appendChild(title);
      for (let idx = 0; idx < rows.length; idx++) {
        const [label, desc] = rows[idx];
        const yy = y + 24 + idx * 20;
        const labelNode = svgEl("text", { x, y: yy, "font-size": 10.5, "font-weight": 700, fill: "#374151" });
        labelNode.textContent = label;
        legend.appendChild(labelNode);
        const descNode = svgEl("text", { x: x + 58, y: yy, "font-size": 10.5, fill: "#64748b" });
        descNode.textContent = desc;
        legend.appendChild(descNode);
      }
      svg.appendChild(legend);
    }

    function shouldShowNetworkLabel(node, graph, showLabels) {
      if (!showLabels) return false;
      if (graph.nodes.length <= 70) return true;
      return Number(node.degree || 0) >= 3 || node.type === "metabolite";
    }

    function addCircularLabel(svg, node, pos, radius, graph, showLabels, dimmed, onClick) {
      if (!shouldShowNetworkLabel(node, graph, showLabels)) return;
      const rightSide = Math.cos(pos.theta) >= 0;
      const label = svgEl("text", {
        x: pos.x + (rightSide ? radius + 8 : -radius - 8),
        y: pos.y + 4,
        "text-anchor": rightSide ? "start" : "end",
        "font-size": node.type === "metabolite" ? 10 : 9,
        fill: dimmed ? "#94a3b8" : "#334155",
        cursor: "pointer"
      });
      label.textContent = node.label;
      label.addEventListener("click", onClick);
      const labelTitle = svgEl("title");
      labelTitle.textContent = nodeDetailText(node);
      label.appendChild(labelTitle);
      svg.appendChild(label);
    }

    function renderNetworkChart(dataset, controls) {
      const width = clamp(Number(controls.width || 1100), 760, 2400);
      const height = clamp(Number(controls.height || 760), 520, 2000);
      const baseNodeSize = clamp(Number(controls.nodeSize || 7), 4, 18);
      const showLabels = Boolean(controls.showLabels);
      const graph = prepareNetworkGraph(dataset, controls);
      if (!graph.edges.length) return null;

      const layout = controls.layout === "cnet" ? "cnet" : "circos";
      if (layout === "cnet") {
        return renderNetworkCnetChart(dataset, controls, graph, width, height, baseNodeSize, showLabels);
      }
      return renderNetworkCircosChart(dataset, controls, graph, width, height, baseNodeSize, showLabels);
    }

    function renderNetworkCircosChart(dataset, controls, graph, width, height, baseNodeSize, showLabels) {
      const cx = width / 2;
      const cy = height / 2 + 18;
      const radius = Math.max(190, Math.min(width - 260, height - 120) / 2);
      const outerR = radius;
      const scaleR = radius / 1.035;
      const radii = {
        outerStripInner: scaleR * 0.992,
        outerStripOuter: scaleR * 1.035,
        trackMeanbarInner: scaleR * 0.86,
        trackMeanbarOuter: scaleR * 0.975,
        trackMeanheatInner: scaleR * 0.795,
        trackMeanheatOuter: scaleR * 0.85,
        trackDegreeInner: scaleR * 0.685,
        trackDegreeOuter: scaleR * 0.775,
        trackCoreInner: scaleR * 0.605,
        trackCoreOuter: scaleR * 0.675,
        trackBiasInner: scaleR * 0.53,
        trackBiasOuter: scaleR * 0.58,
        linkRadius: scaleR * 0.47
      };
      const layoutMap = computeCircosLayout(graph.genes, graph.metabolites);
      const positions = new Map();
      for (const [nodeId, geometry] of layoutMap.entries()) {
        const xy = polarPoint(cx, cy, outerR + 4, geometry.thetaMid);
        positions.set(nodeId, { ...geometry, x: xy.x, y: xy.y, theta: geometry.thetaMid });
      }
      const selection = networkSelectionState(graph, positions, controls);
      const maxDegree = Math.max(1, ...graph.nodes.map(node => Number(node.weightedDegree || node.degree || 0)));
      const maxCore = Math.max(1e-6, ...graph.nodes.map(node => Number(node.moduleCore || 0)).filter(Number.isFinite));
      const maxAbs = Number(dataset.summary?.maxAbsWeight || 1);
      const trackScales = dataset.trackScales || {};

      const svg = svgEl("svg", {
        width,
        height,
        viewBox: `0 0 ${width} ${height}`,
        role: "img",
        "aria-label": dataset.title || "Network Explorer"
      });
      svg.appendChild(svgEl("rect", { x: 0, y: 0, width, height, fill: "#ffffff" }));
      addNetworkTitle(svg, dataset, graph, "Circos layout", width);

      for (const edge of [...graph.edges].sort((a, b) => Number(a.absWeight || 0) - Number(b.absWeight || 0))) {
        const source = layoutMap.get(edge.source);
        const target = layoutMap.get(edge.target);
        if (!source || !target) continue;
        const connectedToSelection = selection.hasSelection && selection.selectedVisible && (edge.source === selection.selectedNodeId || edge.target === selection.selectedNodeId);
        const dimmed = selection.hasSelection && selection.selectedVisible && !connectedToSelection;
        const absWeight = Number(edge.absWeight || Math.abs(edge.edgeWeight || 0));
        const strokeWidth = 0.35 + 3.2 * Math.sqrt(Math.min(1, absWeight / Math.max(1e-6, maxAbs)));
        const path = svgEl("path", {
          d: chordPath(cx, cy, source.thetaMid, target.thetaMid, radii.linkRadius),
          fill: "none",
          stroke: edgeColor(edge),
          "stroke-width": connectedToSelection ? strokeWidth + 1.2 : strokeWidth,
          opacity: dimmed ? 0.05 : connectedToSelection ? 0.82 : 0.40,
          "stroke-linecap": "round"
        });
        const lineTitle = svgEl("title");
        lineTitle.textContent = edgeDetailText(edge);
        path.appendChild(lineTitle);
        svg.appendChild(path);
      }

      for (const node of graph.nodes) {
        const geometry = layoutMap.get(node.id);
        if (!geometry) continue;
        const isSelected = node.id === selection.selectedNodeId;
        const isNeighbor = selection.hasSelection && selection.selectedVisible && selection.neighborIds.has(node.id);
        const dimmed = selection.hasSelection && selection.selectedVisible && !isSelected && !isNeighbor;
        const outerSegment = svgEl("path", {
          d: annularPath(cx, cy, radii.outerStripInner, radii.outerStripOuter, geometry.thetaStart, geometry.thetaEnd),
          fill: nodeColor(node),
          opacity: dimmed ? 0.22 : 1,
          stroke: isSelected ? "#f59e0b" : isNeighbor ? "#fbbf24" : "#ffffff",
          "stroke-width": isSelected ? 2.4 : isNeighbor ? 1.9 : 0.6,
          cursor: "pointer"
        });
        outerSegment.dataset.nodeId = node.id;
        outerSegment.addEventListener("click", event => {
          event.stopPropagation();
          setControl("network_explorer", "selectedNodeId", node.id === selection.selectedNodeId ? "" : node.id);
        });
        const titleNode = svgEl("title");
        titleNode.textContent = nodeDetailText(node);
        outerSegment.appendChild(titleNode);
        svg.appendChild(outerSegment);

        const track2Values = Array.isArray(node.track2Values) ? node.track2Values.map(Number).filter(Number.isFinite) : [];
        const track2Scale = node.type === "gene" ? Number(trackScales.geneTrack2 || trackScales.geneMean || 1) : Number(trackScales.metaboliteTrack2 || trackScales.metaboliteMean || 1);
        svg.appendChild(svgEl("path", {
          d: annularPath(cx, cy, radii.trackMeanbarInner, radii.trackMeanbarOuter, geometry.thetaStart, geometry.thetaEnd),
          fill: "#fbfbfb",
          opacity: dimmed ? 0.18 : 1,
          stroke: "#eef2f7",
          "stroke-width": 0.25
        }));
        if (track2Values.length > 1) {
          const groupColors = dataset.track2?.group1Colors || {};
          const groupOrder = dataset.track2?.group1Order || [];
          for (let idx = 0; idx < track2Values.length; idx++) {
            const value = track2Values[idx];
            const theta = (geometry.thetaStart + geometry.thetaEnd) / 2 + (idx - (track2Values.length - 1) / 2) * Math.max(0.002, (geometry.thetaEnd - geometry.thetaStart) * 0.10);
            const rMid = 0.5 * (radii.trackMeanbarInner + radii.trackMeanbarOuter);
            const radialHalf = 0.42 * (radii.trackMeanbarOuter - radii.trackMeanbarInner);
            const r = rMid + clamp(value / Math.max(1e-6, track2Scale), -1, 1) * radialHalf;
            const xy = polarPoint(cx, cy, r, theta);
            svg.appendChild(svgEl("circle", {
              cx: xy.x,
              cy: xy.y,
              r: 2.2,
              fill: groupColors[groupOrder[idx]] || "#6b7280",
              opacity: dimmed ? 0.18 : 0.92,
              stroke: "none"
            }));
          }
        } else {
          const value = track2Values.length ? track2Values[0] : Number(node.meanZScore || 0);
          const rMid = 0.5 * (radii.trackMeanbarInner + radii.trackMeanbarOuter);
          const rOuter = value >= 0
            ? rMid + clamp(value / Math.max(1e-6, track2Scale), 0, 1) * (radii.trackMeanbarOuter - rMid)
            : rMid + clamp(value / Math.max(1e-6, track2Scale), -1, 0) * (rMid - radii.trackMeanbarInner);
          svg.appendChild(svgEl("path", {
            d: annularPath(cx, cy, Math.min(rMid, rOuter), Math.max(rMid, rOuter), geometry.thetaStart, geometry.thetaEnd),
            fill: "#6b7280",
            opacity: dimmed ? 0.12 : 0.88,
            stroke: "none"
          }));
        }

        const meanScale = node.type === "gene" ? Number(trackScales.geneMean || 1) : Number(trackScales.metaboliteMean || 1);
        svg.appendChild(svgEl("path", {
          d: annularPath(cx, cy, radii.trackMeanheatInner, radii.trackMeanheatOuter, geometry.thetaStart, geometry.thetaEnd),
          fill: signedHeatColor(Number(node.meanZScore || 0), meanScale),
          opacity: dimmed ? 0.15 : 1,
          stroke: "#ffffff",
          "stroke-width": 0.25
        }));

        const degreeScale = node.type === "gene" ? Number(trackScales.geneDegree || maxDegree) : Number(trackScales.metaboliteDegree || maxDegree);
        const degreeOuterR = radii.trackDegreeInner + (radii.trackDegreeOuter - radii.trackDegreeInner) * Math.min(1, Number(node.weightedDegree || 0) / Math.max(1e-6, degreeScale));
        svg.appendChild(svgEl("path", {
          d: annularPath(cx, cy, radii.trackDegreeInner, degreeOuterR, geometry.thetaStart, geometry.thetaEnd),
          fill: "#4b5563",
          opacity: dimmed ? 0.15 : 0.92,
          stroke: "none"
        }));

        const coreScale = node.type === "gene" ? Number(trackScales.geneCore || maxCore) : Number(trackScales.metaboliteCore || maxCore);
        const coreOuterR = radii.trackCoreInner + (radii.trackCoreOuter - radii.trackCoreInner) * Math.min(1, Number(node.moduleCore || 0) / Math.max(1e-6, coreScale));
        svg.appendChild(svgEl("path", {
          d: annularPath(cx, cy, radii.trackCoreInner, coreOuterR, geometry.thetaStart, geometry.thetaEnd),
          fill: node.type === "gene" ? (node.moduleColor || "#9ca3af") : "#8c6d46",
          opacity: dimmed ? 0.14 : 0.92,
          stroke: "none"
        }));

        svg.appendChild(svgEl("path", {
          d: annularPath(cx, cy, radii.trackBiasInner, radii.trackBiasOuter, geometry.thetaStart, geometry.thetaEnd),
          fill: biasColor(node),
          opacity: dimmed ? 0.15 : 0.95,
          stroke: "#ffffff",
          "stroke-width": 0.25
        }));

        const pos = positions.get(node.id);
        if (pos) {
          addCircularLabel(svg, node, pos, baseNodeSize, graph, showLabels, dimmed, event => {
            event.stopPropagation();
            setControl("network_explorer", "selectedNodeId", node.id === selection.selectedNodeId ? "" : node.id);
          });
        }
      }

      const geneLabelPos = polarPoint(cx, cy, outerR + 36, Math.PI * 1.18);
      const geneLabel = svgEl("text", { x: geneLabelPos.x, y: geneLabelPos.y, "text-anchor": "middle", "font-size": 12, "font-weight": 700, fill: "#334155" });
      geneLabel.textContent = "Genes";
      svg.appendChild(geneLabel);
      const metabLabelPos = polarPoint(cx, cy, outerR + 36, Math.PI * 0.06);
      const metabLabel = svgEl("text", { x: metabLabelPos.x, y: metabLabelPos.y, "text-anchor": "middle", "font-size": 12, "font-weight": 700, fill: "#334155" });
      metabLabel.textContent = "Metabolites";
      svg.appendChild(metabLabel);
      addNetworkLegend(svg, 24, 88, "circos");
      addTrackAnnotationLegend(svg, 24, 206);

      svg.addEventListener("click", () => {
        if (getViewControls("network_explorer").selectedNodeId) {
          setControl("network_explorer", "selectedNodeId", "");
        }
      });

      return { svg, graph, layout: "circos" };
    }

    function renderNetworkCnetChart(dataset, controls, graph, width, height, baseNodeSize, showLabels) {
      const cx = width / 2;
      const cy = height / 2 + 18;
      const ringR = Math.max(190, Math.min(width - 250, height - 130) / 2);
      const layoutMap = computeCircosLayout(graph.genes, graph.metabolites);
      const positions = new Map();
      const maxDegree = Math.max(1, ...graph.nodes.map(node => Number(node.degree || 0)));
      for (const node of graph.nodes) {
        const geometry = layoutMap.get(node.id);
        if (!geometry) continue;
        const jitter = 18 * Math.sin((positions.size + 1) * 1.71);
        const xy = polarPoint(cx, cy, ringR + jitter, geometry.thetaMid);
        positions.set(node.id, { ...geometry, x: xy.x, y: xy.y, theta: geometry.thetaMid });
      }
      const selection = networkSelectionState(graph, positions, controls);
      const maxAbs = Number(dataset.summary?.maxAbsWeight || 1);

      const svg = svgEl("svg", {
        width,
        height,
        viewBox: `0 0 ${width} ${height}`,
        role: "img",
        "aria-label": dataset.title || "Network Explorer"
      });
      svg.appendChild(svgEl("rect", { x: 0, y: 0, width, height, fill: "#ffffff" }));
      addNetworkTitle(svg, dataset, graph, "CNet circular layout", width);

      for (const edge of [...graph.edges].sort((a, b) => Number(a.absWeight || 0) - Number(b.absWeight || 0))) {
        const source = positions.get(edge.source);
        const target = positions.get(edge.target);
        if (!source || !target) continue;
        const connectedToSelection = selection.hasSelection && selection.selectedVisible && (edge.source === selection.selectedNodeId || edge.target === selection.selectedNodeId);
        const dimmed = selection.hasSelection && selection.selectedVisible && !connectedToSelection;
        const absWeight = Number(edge.absWeight || Math.abs(edge.edgeWeight || 0));
        const strokeWidth = 0.30 + 2.5 * Math.sqrt(Math.min(1, absWeight / Math.max(1e-6, maxAbs)));
        const path = svgEl("path", {
          d: pointChordPath(source, target, cx, cy, 0.18),
          fill: "none",
          stroke: edge.metaboliteColor || edgeColor(edge),
          "stroke-width": connectedToSelection ? strokeWidth + 1.1 : strokeWidth,
          opacity: dimmed ? 0.06 : connectedToSelection ? 0.86 : 0.56,
          "stroke-linecap": "round"
        });
        const title = svgEl("title");
        title.textContent = edgeDetailText(edge);
        path.appendChild(title);
        svg.appendChild(path);
      }

      for (const node of graph.nodes) {
        const pos = positions.get(node.id);
        if (!pos) continue;
        const isSelected = node.id === selection.selectedNodeId;
        const isNeighbor = selection.hasSelection && selection.selectedVisible && selection.neighborIds.has(node.id);
        const dimmed = selection.hasSelection && selection.selectedVisible && !isSelected && !isNeighbor;
        const radius = baseNodeSize + Math.min(13, Math.sqrt(Number(node.degree || 1) / maxDegree) * 13);
        const circle = svgEl("circle", {
          cx: pos.x,
          cy: pos.y,
          r: isSelected ? radius + 3 : radius,
          fill: nodeColor(node),
          opacity: dimmed ? 0.24 : 0.97,
          stroke: isSelected ? "#f59e0b" : isNeighbor ? "#fbbf24" : "#ffffff",
          "stroke-width": isSelected ? 3 : isNeighbor ? 2.2 : 1.1,
          cursor: "pointer"
        });
        circle.dataset.nodeId = node.id;
        circle.addEventListener("click", event => {
          event.stopPropagation();
          setControl("network_explorer", "selectedNodeId", node.id === selection.selectedNodeId ? "" : node.id);
        });
        const nodeTitle = svgEl("title");
        nodeTitle.textContent = nodeDetailText(node);
        circle.appendChild(nodeTitle);
        svg.appendChild(circle);

        addCircularLabel(svg, node, pos, radius, graph, showLabels, dimmed, event => {
          event.stopPropagation();
          setControl("network_explorer", "selectedNodeId", node.id === selection.selectedNodeId ? "" : node.id);
        });
      }

      addNetworkLegend(svg, 24, 88, "cnet");
      svg.addEventListener("click", () => {
        if (getViewControls("network_explorer").selectedNodeId) {
          setControl("network_explorer", "selectedNodeId", "");
        }
      });

      return { svg, graph, layout: "cnet" };
    }

    function renderNetworkSummary(dataset, rendered) {
      const legend = el("div", { className: "legend" });
      legend.appendChild(el("span", { className: "legend-item", text: "Source: T02 high-confidence network" }));
      legend.appendChild(el("span", { className: "legend-item", text: `Edges: ${rendered.graph.edges.length}/${(dataset.edges || []).length}` }));
      legend.appendChild(el("span", { className: "legend-item", text: `Genes shown: ${rendered.graph.genes.length}/${dataset.summary?.genes || 0}` }));
      legend.appendChild(el("span", { className: "legend-item", text: `Metabolites shown: ${rendered.graph.metabolites.length}/${dataset.summary?.metabolites || 0}` }));
      if (getViewControls("network_explorer").selectedNodeId) {
        const node = rendered.graph.nodeLookup.get(getViewControls("network_explorer").selectedNodeId);
        if (node) legend.appendChild(el("span", { className: "legend-item", text: `Selected: ${node.label}` }));
      }
      return legend;
    }

    function renderPcaView(view) {
      const dataset = getActiveDataset();
      const controls = getViewControls(view.id);
      const panel = el("section", { className: "panel" });
      const title = dataset ? dataset.title : view.title;
      panel.appendChild(el("div", { className: "panel-head" }, [
        el("h2", { className: "panel-title", text: title }),
        el("p", {
          className: "panel-note",
          text: "Switch dataset, inspect samples by hover, and export the current SVG snapshot."
        })
      ]));

      const schema = report.schemas[view.schema_id];
      const controlsRow = el("div", { className: "controls" });
      for (const control of schema.controls || []) {
        controlsRow.appendChild(renderControlField(view, control));
      }
      panel.appendChild(controlsRow);

      const actionBar = el("div", { className: "action-bar" }, [
        el("button", { text: "Export SVG", onclick: () => downloadSvg(chartShell.querySelector("svg"), `${dataset ? dataset.id : "pca"}.svg`) }),
        el("button", { text: "Reset", onclick: () => resetControls(view.id) })
      ]);
      panel.appendChild(actionBar);

      const chartWrap = el("div", { className: "chart-wrap" });
      const chartShell = el("div", { className: "chart-shell" });
      if (dataset) {
        chartShell.appendChild(renderPcaChart(dataset, controls));
        chartWrap.appendChild(chartShell);
        panel.appendChild(chartWrap);
        panel.appendChild(renderPcaLegend(dataset, controls.colorBy));
      } else {
        chartWrap.appendChild(el("div", { className: "placeholder", text: "No PCA payload available for the selected dataset." }));
        panel.appendChild(chartWrap);
      }
      return panel;
    }

    function renderAssociationView(view) {
      const dataset = getAssociationDataset();
      const controls = getViewControls(view.id);
      if (dataset) {
        resolveAssociationControlDefaults(dataset, controls);
      }

      const panel = el("section", { className: "panel" });
      panel.appendChild(el("div", { className: "panel-head" }, [
        el("h2", { className: "panel-title", text: dataset ? `${dataset.title}` : view.title }),
        el("p", {
          className: "panel-note",
          text: "Switch between gene-metabolite and module-metabolite pairs. Scatter points and confidence bands use the associated module color."
        })
      ]));

      const schema = report.schemas[view.schema_id];
      const controlsRow = el("div", { className: "controls" });
      for (const control of schema.controls || []) {
        controlsRow.appendChild(renderControlField(view, control));
      }
      panel.appendChild(controlsRow);

      const actionBar = el("div", { className: "action-bar" }, [
        el("button", { text: "Export SVG", onclick: () => downloadSvg(chartShell.querySelector("svg"), `${dataset ? dataset.id : "association"}.svg`) }),
        el("button", { text: "Reset", onclick: () => resetControls(view.id) })
      ]);
      panel.appendChild(actionBar);

      const chartWrap = el("div", { className: "chart-wrap" });
      const chartShell = el("div", { className: "chart-shell" });
      if (dataset) {
        const rendered = renderAssociationChart(dataset, controls);
        if (rendered && rendered.svg) {
          chartShell.appendChild(rendered.svg);
          chartWrap.appendChild(chartShell);
          panel.appendChild(chartWrap);
          panel.appendChild(rendered.summary);
        } else {
          chartWrap.appendChild(el("div", { className: "placeholder", text: "No valid association payload available." }));
          panel.appendChild(chartWrap);
        }
      } else {
        chartWrap.appendChild(el("div", { className: "placeholder", text: "No regression payload available for the selected type." }));
        panel.appendChild(chartWrap);
      }
      return panel;
    }

    function renderModuleHeatmapView(view) {
      const dataset = report.datasets.module_heatmap || null;
      const controls = getViewControls(view.id);
      const panel = el("section", { className: "panel" });
      panel.appendChild(el("div", { className: "panel-head" }, [
        el("h2", { className: "panel-title", text: dataset ? dataset.title : view.title }),
        el("p", {
          className: "panel-note",
          text: "Filter modules and metabolites, sort rows and columns, and export the current Spearman association heatmap."
        })
      ]));

      const schema = report.schemas[view.schema_id];
      const controlsRow = el("div", { className: "controls" });
      for (const control of schema.controls || []) {
        controlsRow.appendChild(renderControlField(view, control));
      }
      panel.appendChild(controlsRow);

      const actionBar = el("div", { className: "action-bar" }, [
        el("button", { text: "Export SVG", onclick: () => {
          const svg = chartShell.querySelector("svg");
          if (svg) downloadSvg(svg, "module_heatmap.svg");
        }}),
        el("button", { text: "Reset", onclick: () => resetControls(view.id) })
      ]);
      panel.appendChild(actionBar);

      const chartWrap = el("div", { className: "chart-wrap" });
      const chartShell = el("div", { className: "chart-shell" });
      if (dataset) {
        const rendered = renderModuleHeatmapChart(dataset, controls);
        if (rendered && rendered.svg) {
          chartShell.appendChild(rendered.svg);
          chartWrap.appendChild(chartShell);
          panel.appendChild(chartWrap);
          panel.appendChild(renderModuleHeatmapSummary(dataset, rendered));
        } else {
          chartWrap.appendChild(el("div", { className: "placeholder", text: "No module-metabolite cells are available for the selected filters." }));
          panel.appendChild(chartWrap);
        }
      } else {
        chartWrap.appendChild(el("div", { className: "placeholder", text: "No module-metabolite association payload available." }));
        panel.appendChild(chartWrap);
      }
      return panel;
    }

    function renderNetworkView(view) {
      const dataset = getNetworkDataset();
      const controls = getViewControls(view.id);
      const panel = el("section", { className: "panel" });
      panel.appendChild(el("div", { className: "panel-head" }, [
        el("h2", { className: "panel-title", text: dataset ? dataset.title : view.title }),
        el("p", {
          className: "panel-note",
          text: "T02-only Circos and CNet views. Inspect nodes by hover and click a node to highlight first-order neighbors."
        })
      ]));

      const schema = report.schemas[view.schema_id];
      const controlsRow = el("div", { className: "controls" });
      for (const control of schema.controls || []) {
        controlsRow.appendChild(renderControlField(view, control));
      }
      panel.appendChild(controlsRow);

      const actionBar = el("div", { className: "action-bar" }, [
        el("button", { text: "Export SVG", onclick: () => {
          const svg = chartShell.querySelector("svg");
          if (svg) downloadSvg(svg, `${dataset ? dataset.id : "network"}.svg`);
        }}),
        el("button", { text: "Reset", onclick: () => resetControls(view.id) })
      ]);
      panel.appendChild(actionBar);

      const chartWrap = el("div", { className: "chart-wrap" });
      const chartShell = el("div", { className: "chart-shell" });
      if (dataset) {
        const rendered = renderNetworkChart(dataset, controls);
        if (rendered && rendered.svg) {
          chartShell.appendChild(rendered.svg);
          chartWrap.appendChild(chartShell);
          panel.appendChild(chartWrap);
          panel.appendChild(renderNetworkSummary(dataset, rendered));
        } else {
          chartWrap.appendChild(el("div", { className: "placeholder", text: "No network edges match the selected filters." }));
          panel.appendChild(chartWrap);
        }
      } else {
        chartWrap.appendChild(el("div", { className: "placeholder", text: "No T02 high-confidence network payload available." }));
        panel.appendChild(chartWrap);
      }
      return panel;
    }

    function renderPlaceholderView(view) {
      const panel = el("section", { className: "panel" });
      panel.appendChild(el("div", { className: "panel-head" }, [
        el("h2", { className: "panel-title", text: view.title }),
        el("p", { className: "panel-note", text: "This view is reserved for a later stage." })
      ]));
      panel.appendChild(el("div", { className: "placeholder", text: view.description || "Not implemented yet." }));
      return panel;
    }

    function renderMain() {
      const main = el("main", { className: "main" });
      const view = getView(state.activeViewId) || report.views[0];
      if (view.kind === "gallery") {
        main.appendChild(renderGalleryView(view));
      } else if (view.kind === "pca") {
        main.appendChild(renderPcaView(view));
      } else if (view.kind === "association") {
        main.appendChild(renderAssociationView(view));
      } else if (view.kind === "module_heatmap") {
        main.appendChild(renderModuleHeatmapView(view));
      } else if (view.kind === "network") {
        main.appendChild(renderNetworkView(view));
      } else {
        main.appendChild(renderPlaceholderView(view));
      }
      return main;
    }

    function render() {
      clear(app);
      app.appendChild(renderSidebar());
      try {
        app.appendChild(renderMain());
      } catch (error) {
        app.appendChild(renderRuntimeError(error));
      }
    }

    render();
  </script>
</body>
</html>
"""



__all__ = [
    "_interactive_html_template",
]

