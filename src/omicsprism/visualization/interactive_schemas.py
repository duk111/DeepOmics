from __future__ import annotations

from typing import Any

def _build_pca_schema(default_dataset: str) -> dict[str, Any]:
    return {
        "id": "pca.scatter",
        "title": "PCA controls",
        "controls": [
            {
                "id": "dataset",
                "type": "select",
                "label": "Dataset",
                "default": default_dataset,
                "options": [
                    {"value": "transcriptome", "label": "Transcriptome"},
                    {"value": "metabolome", "label": "Metabolome"},
                ],
            },
            {
                "id": "colorBy",
                "type": "select",
                "label": "Color",
                "default": "group1",
                "options": [
                    {"value": "group1", "label": "Group 1"},
                    {"value": "group2", "label": "Group 2"},
                ],
            },
            {
                "id": "xComponent",
                "type": "select",
                "label": "X component",
                "default": 1,
                "options": [
                    {"value": 1, "label": "PC1"},
                    {"value": 2, "label": "PC2"},
                    {"value": 3, "label": "PC3"},
                    {"value": 4, "label": "PC4"},
                    {"value": 5, "label": "PC5"},
                ],
            },
            {
                "id": "yComponent",
                "type": "select",
                "label": "Y component",
                "default": 2,
                "options": [
                    {"value": 1, "label": "PC1"},
                    {"value": 2, "label": "PC2"},
                    {"value": 3, "label": "PC3"},
                    {"value": 4, "label": "PC4"},
                    {"value": 5, "label": "PC5"},
                ],
            },
            {
                "id": "pointSize",
                "type": "range",
                "label": "Point size",
                "default": 5,
                "min": 2,
                "max": 14,
                "step": 0.5,
            },
            {
                "id": "showLabels",
                "type": "toggle",
                "label": "Labels",
                "default": False,
            },
            {
                "id": "width",
                "type": "number",
                "label": "Width",
                "default": 900,
                "min": 640,
                "max": 1800,
                "step": 20,
            },
            {
                "id": "height",
                "type": "number",
                "label": "Height",
                "default": 620,
                "min": 480,
                "max": 1400,
                "step": 20,
            },
        ],
    }


def _build_association_schema(default_pair_type: str, default_gene: str, default_metabolite: str) -> dict[str, Any]:
    return {
        "id": "association.scatter",
        "title": "Association Scatter Studio",
        "controls": [
            {
                "id": "pairType",
                "type": "select",
                "label": "Type",
                "default": default_pair_type,
                "options": [
                    {"value": "gene_metabolite", "label": "Gene-metabolite"},
                    {"value": "module_metabolite", "label": "Module-metabolite"},
                ],
            },
            {
                "id": "topEdgeId",
                "type": "select",
                "label": "Pair",
                "default": "",
                "optionsSource": "topEdges",
                "allowEmpty": True,
                "emptyLabel": "Custom pair",
            },
            {
                "id": "gene",
                "type": "select",
                "label": "Gene / module",
                "default": default_gene,
                "optionsSource": "geneOptions",
            },
            {
                "id": "metabolite",
                "type": "select",
                "label": "Metabolite",
                "default": default_metabolite,
                "optionsSource": "metaboliteOptions",
            },
            {
                "id": "pointSize",
                "type": "range",
                "label": "Point size",
                "default": 5,
                "min": 2,
                "max": 14,
                "step": 0.5,
            },
            {
                "id": "alpha",
                "type": "range",
                "label": "Opacity",
                "default": 0.85,
                "min": 0.15,
                "max": 1.0,
                "step": 0.05,
            },
            {
                "id": "showLabels",
                "type": "toggle",
                "label": "Labels",
                "default": False,
            },
            {
                "id": "showRegression",
                "type": "toggle",
                "label": "Regression line",
                "default": True,
            },
            {
                "id": "width",
                "type": "number",
                "label": "Width",
                "default": 900,
                "min": 640,
                "max": 2000,
                "step": 20,
            },
            {
                "id": "height",
                "type": "number",
                "label": "Height",
                "default": 640,
                "min": 480,
                "max": 1800,
                "step": 20,
            },
        ],
    }


def _build_module_heatmap_schema(default_top_modules: int, default_top_metabolites: int) -> dict[str, Any]:
    return {
        "id": "module.heatmap",
        "title": "Module Heatmap Studio",
        "controls": [
            {
                "id": "topModules",
                "type": "number",
                "label": "Top modules",
                "default": int(default_top_modules),
                "min": 1,
                "max": 200,
                "step": 1,
            },
            {
                "id": "topMetabolites",
                "type": "number",
                "label": "Top metabolites",
                "default": int(default_top_metabolites),
                "min": 1,
                "max": 300,
                "step": 1,
            },
            {
                "id": "palette",
                "type": "select",
                "label": "Palette",
                "default": "rdbu",
                "options": [
                    {"value": "rdbu", "label": "Red-Blue"},
                    {"value": "blueorange", "label": "Blue-Orange"},
                    {"value": "purplegreen", "label": "Purple-Green"},
                ],
            },
            {
                "id": "showValues",
                "type": "toggle",
                "label": "Values",
                "default": False,
            },
            {
                "id": "showStars",
                "type": "toggle",
                "label": "Stars",
                "default": True,
            },
            {
                "id": "rowSort",
                "type": "select",
                "label": "Rows",
                "default": "default",
                "options": [
                    {"value": "default", "label": "Module summary order"},
                    {"value": "max_abs_rho", "label": "Max |rho|"},
                    {"value": "significance", "label": "Significance"},
                    {"value": "name", "label": "Name"},
                ],
            },
            {
                "id": "columnSort",
                "type": "select",
                "label": "Columns",
                "default": "significance",
                "options": [
                    {"value": "significance", "label": "Significance"},
                    {"value": "max_abs_rho", "label": "Max |rho|"},
                    {"value": "name", "label": "Name"},
                    {"value": "default", "label": "Default"},
                ],
            },
            {
                "id": "width",
                "type": "number",
                "label": "Width",
                "default": 980,
                "min": 720,
                "max": 2400,
                "step": 20,
            },
            {
                "id": "height",
                "type": "number",
                "label": "Height",
                "default": 720,
                "min": 520,
                "max": 2000,
                "step": 20,
            },
        ],
    }


def _build_network_schema(default_top_edges: int) -> dict[str, Any]:
    return {
        "id": "network.explorer",
        "title": "Network Explorer",
        "controls": [
            {
                "id": "layout",
                "type": "select",
                "label": "Layout",
                "default": "circos",
                "options": [
                    {"value": "circos", "label": "Circos"},
                    {"value": "cnet", "label": "CNet"},
                ],
            },
            {
                "id": "nodeSize",
                "type": "range",
                "label": "Node size",
                "default": 7,
                "min": 4,
                "max": 18,
                "step": 0.5,
            },
            {
                "id": "showLabels",
                "type": "toggle",
                "label": "Labels",
                "default": False,
            },
            {
                "id": "width",
                "type": "number",
                "label": "Width",
                "default": 1100,
                "min": 760,
                "max": 2400,
                "step": 20,
            },
            {
                "id": "height",
                "type": "number",
                "label": "Height",
                "default": 760,
                "min": 520,
                "max": 2000,
                "step": 20,
            },
        ],
    }


def _build_upset_schema(default_max_intersections: int) -> dict[str, Any]:
    return {
        "id": "upset.overlap",
        "title": "UpSet Explorer",
        "controls": [
            {
                "id": "sortBy",
                "type": "select",
                "label": "Sort by",
                "default": "size",
                "options": [
                    {"value": "size", "label": "Size"},
                    {"value": "degree", "label": "Support degree"},
                    {"value": "combination", "label": "Combination"},
                ],
            },
            {
                "id": "maxIntersections",
                "type": "select",
                "label": "Max intersections",
                "default": int(default_max_intersections),
                "options": [
                    {"value": 10, "label": "10"},
                    {"value": 20, "label": "20"},
                    {"value": 30, "label": "30"},
                    {"value": 40, "label": "40"},
                    {"value": 50, "label": "50"},
                ],
            },
            {
                "id": "zoom",
                "type": "range",
                "label": "Zoom",
                "default": 1.0,
                "min": 0.5,
                "max": 3.0,
                "step": 0.1,
            },
            {
                "id": "width",
                "type": "number",
                "label": "Width",
                "default": 1120,
                "min": 760,
                "max": 2400,
                "step": 20,
            },
            {
                "id": "height",
                "type": "number",
                "label": "Height",
                "default": 680,
                "min": 520,
                "max": 1600,
                "step": 20,
            },
        ],
    }


def _build_placeholder_schema(schema_id: str, title: str) -> dict[str, Any]:
    return {
        "id": schema_id,
        "title": title,
        "controls": [],
    }



__all__ = [
    "_build_association_schema",
    "_build_module_heatmap_schema",
    "_build_network_schema",
    "_build_pca_schema",
    "_build_placeholder_schema",
    "_build_upset_schema",
]
