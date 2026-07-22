# OmicsPrism 项目上下文说明

本文档用于让 AI 在不读取源码的情况下理解 OmicsPrism 项目现状，包括项目目标、输入输出、核心流程、结果文件和可视化能力。

## 1. 项目一句话

OmicsPrism 是一个面向转录组和代谢组联合分析的 Python 工具包。它接收基因表达矩阵、代谢物矩阵和样本分组表，完成多组学数据预处理、基因-代谢物关联建模、关键基因筛选、关联网络构建、基因模块检测，并自动导出结构化结果表、静态论文图和浏览器原生交互报告。

## 2. 项目定位

OmicsPrism 的核心目标不是做通用绘图库，而是把“转录组-代谢组关联分析”沉淀成一个可重复运行的端到端分析流水线。

主要使用者包括：

- 生物信息分析人员：批量运行多组学关联分析，得到候选关键基因、代谢物关联网络和模块结果。
- 课题研究人员：查看 PCA、回归散点图、模块热图、Circos 网络图、UpSet 图等解释性图形。

项目当前已经具备命令行入口、Python API、结果表导出、静态图导出和交互 HTML 报告。

## 3. 当前运行方式

安装方式：

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
python -m pip install -e .
```

典型命令：

```bash
omicsprism run --genes DEAT.csv --metabs ym_metab.csv --group-table group.csv --output results/DEAT_results_3
```

CLI 入口为 `omicsprism run`。关键参数包括：

- `--genes`：转录组矩阵 CSV，格式为 features x samples，即行是基因，列是样本。
- `--metabs`：代谢组矩阵 CSV，格式为 features x samples，即行是代谢物，列是样本。
- `--group-table`：样本分组表，必须包含 `sample_id`、`group1`、`group2` 三列。
- `--output`：结果输出目录。
- `--trans-log2 / --no-trans-log2`：是否对转录组做 `log2(x+1)`。如果输入已经是 VST 或其他已变换矩阵，应关闭。
- `--enable-modules / --disable-modules`：是否启用基因模块分析。
- `--report-format md|html`：报告格式。HTML 报告会同时生成交互式可视化报告。

### 3.1 可选前置模块：差异表达分析

如果用户手里是 RNA-seq raw count 矩阵和样本分组元数据，而不是已经筛选或变换后的基因表达矩阵，推荐先运行独立的差异分析模块：

```bash
omicsprism deg \
  --counts raw_count.csv \
  --metadata metadata.csv \
  --out results/deg \
  --same-fields line,timepoint \
  --compare-field treatment \
  --tested-levels salt \
  --reference-level control
```

`omicsprism deg` 与主流程 `omicsprism run` 是两个独立命令。差异分析模块只负责从 raw count 和 metadata 生成差异基因结果，不会启动代谢组关联建模，也不会改变原有 OmicsPrism 主流程。

差异分析模块会输出：

- `{contrast_name}.all.csv`：每个比较的全量差异分析结果表，包含显著和非显著基因，并增加 `volcano_status` 字段用于火山图分组。
- `{contrast_name}.sig.csv`：每个比较的显著差异基因表。
- `differential_gene_counts.csv`：每个比较的上调基因数、下调基因数、显著基因总数、非显著基因数和总基因数。
- `union_significant_genes.csv`：所有比较的显著基因并集摘要。
- `union_significant_genes.vst.csv`：显著基因并集的 VST 表达矩阵，格式为 genes x samples，可作为 `omicsprism run --genes` 的输入。
- `plots/volcano/{contrast_name}.volcano.png`：每个比较的火山图。横轴为 `log2 Fold Change`，纵轴为 `-log10 adjusted P value`，颜色区分上调基因、下调基因和非显著基因。
- `plots/ma/{contrast_name}.ma.png`：每个比较的 MA 图。横轴为 `log2(baseMean + 1)`，其中 `baseMean` 是 DESeq2 结果中的 normalized count 均值；纵轴为 `log2 Fold Change`，每个点代表一个基因。
- `plots/deg_counts/differential_gene_counts.bar.png`：差异基因数量柱状图。上调基因数量向上，下调基因数量以负值向下。
- `plots/upset/differential_gene_upset.png`：不同比较之间显著差异基因的交集 UpSet 图。
- `plots/sankey/differential_sankey.html|png`：当设置了 `--same-fields` 时生成，按用户传入的 same-fields 顺序展示各层级到 Up 和 Down count 的流向；同一层同名节点会合并，节点从上到下按 metadata 对应列的首次出现顺序排列，不同层和方向使用不同颜色；当 tested levels 多于一个时，会在末端方向前增加 tested-vs-reference 层。

随后使用差异基因 VST 矩阵进入原 OmicsPrism 主流程：

```bash
omicsprism run \
  --genes results/deg/union_significant_genes.vst.csv \
  --metabs ym_metab.csv \
  --group-table group.csv \
  --output results/omicsprism_after_deg \
  --no-trans-log2
```

这里必须使用 `--no-trans-log2`，因为 `union_significant_genes.vst.csv` 已经是 VST 变换后的表达矩阵，不应再做 `log2(x+1)`。

差异分析模块依赖 PyDESeq2。OmicsPrism 主流程仍保持原依赖不变；由于当前 PyDESeq2 版本要求 Python 3.11+，如果需要使用该模块，请在 Python 3.11+ 环境中安装可选依赖：

```bash
python -m pip install -e .[deg]
```

### 3.2 可选前置模块：差异代谢物分析

如果用户希望先基于分组比较筛选差异代谢物，可以运行独立的 `dem` 模块。该模块结合 OPLS-DA 模型 VIP 值、fold change 和 Welch t 检验 P 值筛选差异代谢物。

```bash
omicsprism dem \
  --metabs ym_metab.csv \
  --metadata metadata.csv \
  --out results/dem \
  --same-fields line,timepoint \
  --compare-field treatment \
  --tested-levels salt \
  --reference-level control
```

默认联合筛选条件为 `VIP >= 1`、`padj_bh <= 0.05`、`|log2FoldChange| >= 1`。可通过 `--vip-cutoff`、`--padj-cutoff` 和 `--log2fc-cutoff` 调整。

`omicsprism dem` 默认按原始代谢峰面积处理：先做缺失率过滤，再用 half-min 填补缺失值，随后做样本级 median normalization 和 log2 转换；FC 基于填补并归一化但未 log 的数据计算，Welch t 检验基于 log2 后的数据计算，OPLS-DA/VIP 基于 log2 后再 Pareto scaling 的数据计算。如果输入已经预处理，可用 `--impute median` 改为中位数填补，用 `--no-normalize` 关闭 median normalization，用 `--no-log` 关闭 log2 转换。

差异代谢物模块会输出：

- `{contrast_name}.all.csv`：每个比较的全量结果表，包含 `vip`、`fold_change`、`log2FoldChange`、`t_stat`、`pvalue`、`padj_bh` 和 `dem_status`。
- `{contrast_name}.sig.csv`：每个比较的显著差异代谢物表。
- `{contrast_name}.oplsda_scores.csv`：每个比较的 OPLS-DA 样本得分。
- `differential_metabolite_counts.csv`：每个比较的上调、下调和总差异代谢物数量。
- `union_significant_metabolites.csv`：所有比较的显著差异代谢物并集摘要。
- `union_significant_metabolites.matrix.csv`：显著差异代谢物并集矩阵，格式为 metabolites x samples，可作为 `omicsprism run --metabs` 的输入。
- `plots/volcano/`、`plots/vip/`、`plots/oplsda_scores/` 和 `plots/dem_counts/`：对应的火山图、VIP 图、OPLS-DA 得分图和差异代谢物数量图。
- `plots/vip_log2fc_padj/`：DEM 的 VIP-log2FC-padj 联合散点图；横轴为 `log2FoldChange`，纵轴为 VIP，点大小按 `-log10(padj_bh)`。
- `plots/padj_log2fc_vip/`：DEM 的 padj-log2FC-VIP 联合散点图；横轴为 `log2FoldChange`，纵轴为 `-log10(padj_bh)`，点大小按 VIP。
- `plots/upset/differential_metabolite_upset.png`：不同比较之间显著差异代谢物的交集 UpSet 图。
- `plots/sankey/differential_sankey.html|png`：当设置了 `--same-fields` 时生成，按用户传入的 same-fields 顺序展示各层级到 Up 和 Down count 的流向；同一层同名节点会合并，节点从上到下按 metadata 对应列的首次出现顺序排列，不同层和方向使用不同颜色；当 tested levels 多于一个时，会在末端方向前增加 tested-vs-reference 层。

## 4. 输入数据契约

### 4.1 转录组矩阵

- CSV 文件。
- 第一列为基因 ID 或基因名，作为行索引。
- 后续列为样本 ID。
- 数值为基因表达量。
- 不允许重复基因 ID。
- 不允许重复样本 ID。
- 非数值会被转为缺失值，并在预处理阶段处理。

示意：

```text
Gene,S1,S2,S3
GeneA,10,12,9
GeneB,4,5,6
```

### 4.2 代谢组矩阵

- CSV 文件。
- 第一列为代谢物 ID 或名称，作为行索引。
- 后续列为样本 ID。
- 数值为代谢物丰度。
- 不允许重复代谢物 ID。
- 不允许重复样本 ID。

示意：

```text
Metabolite,S1,S2,S3
M1,100,120,90
M2,40,50,60
```

### 4.3 样本分组表

分组表必须包含三列：

- `sample_id`：样本 ID，必须能与转录组和代谢组矩阵列名匹配。
- `group1`：一级分组，常用于主分组着色、PCA 分组展示、折线图分面等。
- `group2`：二级分组，常用于重复测量、子分组、热图注释、PCA 子分组展示。

示意：

```text
sample_id,group1,group2
S1,Control,T1
S2,Control,T1
S3,Treatment,T1
```

重要业务规则：

- 样本顺序优先按分组表中的 `sample_id` 顺序对齐。
- 仅保留转录组和代谢组共同拥有的样本。
- 关联建模时会按 `group1 + group2` 对重复测量样本取均值。
- PCA 和样本聚类图使用未取均值的原始样本视角，以保留重复样本分布。

## 5. 内部数据模型

项目使用 AnnData 作为核心容器。

主要字段语义：

- `adata.X`：转录组矩阵，样本 x 基因。
- `adata.obs_names`：样本 ID。
- `adata.var_names`：基因 ID。
- `adata.obsm["metabolomics"]`：代谢组矩阵，样本 x 代谢物。
- `adata.obsm["metabolomics_scaled"]`：标准化后的代谢组矩阵。
- `adata.uns["metabolite_names"]`：代谢物名称列表。
- `adata.layers["raw"]`：标准化前的转录组矩阵副本。
- `adata.uns["input_summary"]`：输入样本、基因、代谢物数量和样本排序信息。
- `adata.uns["preprocess_summary"]`：预处理摘要。

核心引擎中同时保留两套样本视角：

- `engine.plot_adata`：未按重复测量取均值的数据，用于 PCA、聚类等样本层面可视化。
- `engine.adata`：按 `group1 + group2` 平均后的数据，用于基因-代谢物关联建模、网络构建和模块分析。

## 6. 分析流水线

端到端流程如下：

1. 读取转录组和代谢组 CSV。
2. 校验空表、重复 ID、数值类型和样本交集。
3. 根据分组表顺序对齐共同样本。
4. 过滤缺失比例过高的特征，默认阈值为 0.5。
5. 对转录组可选执行 `log2(x+1)`，对代谢组执行 `log2(x+1)`。
6. 使用 KNNImputer 填补缺失值，默认邻居数为 5。
7. 去除零方差基因和零方差代谢物。
8. 对转录组和代谢组做 z-score 标准化。
9. 按 `group1 + group2` 对重复测量样本取均值，用于关联建模。
10. 对每个代谢物执行三路候选基因筛选。
11. 对候选基因执行 ElasticNet 和 XGBoost 关联建模。
12. 使用 RRA 聚合模型排序并计算边权重。
13. 构建总关联网络和高置信关联网络。
14. 汇总关键基因、代谢物关联摘要。
15. 对高置信网络中的基因做模块检测。
16. 计算模块特征向量、模块-代谢物相关性、模块内基因指标。
17. 导出 CSV 结果表、Cytoscape 网络表、分析元数据和 H5AD 状态文件。
18. 生成静态图、普通 HTML 报告和交互式 HTML 报告。

## 7. 关联建模方法

### 7.1 三路候选筛选

每个代谢物会先从全基因集合中筛出候选基因。筛选证据包括：

- Pearson 相关。
- Spearman 相关。
- Mutual Information。

每种方法保留前 `screen_top_k_per_method` 个候选，默认 1000。候选基因表会记录：

- `In_PCC`
- `In_Spearman`
- `In_MI`
- `ScreenSupportCount`
- 相关系数、P 值、FDR、MI 分数等。

### 7.2 机器学习模型

候选基因进入两个模型：

- ElasticNet：线性稀疏回归，适合发现线性贡献。
- XGBoost：非线性树模型，适合捕捉复杂关系。

模型输出包括：

- `ElasticNetScore`
- `ElasticNetRank`
- `ElasticNetSelected`
- `XGBoostScore`
- `XGBoostRank`
- `XGBoostSelected`
- `ModelSupportCount`

### 7.3 RRA 聚合和边权重

项目使用 RRA 思路聚合 ElasticNet 和 XGBoost 排名，并计算网络边权重。

边权重由多类证据综合：

- RRA 排名权重。
- Pearson/Spearman 相关强度。
- ElasticNet/XGBoost 模型支持数。
- 三路筛选支持数。

最终形成：

- `RRAScore`
- `RRARank`
- `RRAWeight`
- `CorrScore`
- `ModelScore`
- `ScreenScore`
- `EdgeWeight`
- `Sign`

## 8. 网络定义

### 8.1 总关联网络

总关联网络来自 ElasticNet top-k 和 XGBoost top-k 的并集，主要保留在内存结果和交互报告数据中，用于证据追溯、UpSet 图和模型审计。默认不再作为主 CSV 表导出，以减少面向生信解释结果的冗余。

### 8.2 高置信网络

高置信网络是总关联网络的严格子集，过滤规则大致为：

- 在目标 top-k 范围内。
- 通过 RRA 排名过滤。
- 至少具备双模型支持，或具备多筛选证据支持。

输出文件为：

```text
T02_High_Confidence_Network.csv
```

后续模块分析和主要网络图通常优先使用高置信网络。

### 8.3 Cytoscape 网络

高置信网络表 `T02_High_Confidence_Network.csv` 已包含 `Source`、`Target` 和 `Interaction` 字段，可直接作为 Cytoscape 边表导入，不再额外导出重复的 Cytoscape 专用表。

## 9. 基因模块分析

模块分析建立在高置信网络相关基因上。

主要步骤：

1. 从高置信网络中提取基因集合。
2. 计算这些基因之间的 Spearman 相关。
3. 构建正相关邻接图。
4. 每个基因保留 top-k 正相关邻居，并按最小边权过滤。
5. 使用 Leiden 或 hierarchical 方法检测模块。
6. 小于 `module_min_size` 的模块合并为 grey。
7. 对每个模块计算 eigengene，即模块特征向量。
8. 计算模块与代谢物之间的 Spearman 相关和显著性。
9. 计算模块内 kME、连接度、关键基因标记等。

关键配置：

- `module_corr_method`：当前支持 `spearman`。
- `module_graph_k`：默认 10。
- `module_min_edge_weight`：默认 0.15。
- `module_method`：`leiden` 或 `hierarchical`。
- `module_resolution`：Leiden 分辨率，默认 1.0。
- `module_min_size`：默认 5。

## 10. 输出文件契约

每次运行会在输出目录生成结构化结果。

### 10.1 结果表

| 文件 | 含义 |
| --- | --- |
| `T01_Metabolite_Association_Summary.csv` | 每个代谢物的三路筛选数量、候选基因数、网络边数、最佳基因等摘要。 |
| `T02_High_Confidence_Network.csv` | 高置信基因-代谢物关联网络，包含 Cytoscape 可用的 `Source`、`Target`、`Interaction` 字段。 |
| `T03_Key_Gene_Summary.csv` | 跨代谢物汇总的关键基因表。 |
| `T04_Gene_Module_Assignment.csv` | 基因所属模块、模块内指标和关键基因注释。 |
| `T05_Module_Metabolite_Association.csv` | 模块与代谢物的 Spearman 相关性和 FDR。 |
| `T06_Module_Summary.csv` | 每个模块的规模、hub 基因、关联代谢物等摘要。 |
| `T99_Metabolite_Gene_Scoring_Audit.csv` | 可选审计表。使用 `--export-audit-tables` 或 `export_audit_tables=True` 时导出，包含完整候选打分、模型和 RRA 字段。 |

### 10.2 状态和元数据

| 文件 | 含义 |
| --- | --- |
| `analysis_metadata.json` | 项目名、配置、样本数、基因数、代谢物数、网络边数、模块数等摘要。 |
| `OmicsPrism_Final_State.h5ad` | 最终 AnnData 状态，便于后续复用。 |
| `omicsprism.log` | 运行日志。 |

### 10.3 报告

| 文件 | 含义 |
| --- | --- |
| `OmicsPrism_Report.html` | 静态 HTML 摘要报告，列出结果表、图文件和部分结果预览。 |
| `OmicsPrism_Report.md` | 可选 Markdown 报告。 |
| `OmicsPrism_Interactive_Report.html` | 浏览器原生交互报告，内嵌数据 payload、控件 schema 和 SVG 绘图逻辑。 |

## 11. 当前静态可视化能力

静态图会输出到：

```text
<output>/plots/
```

多数图支持 `pdf`、`svg`、`png` 三种格式。

| 图文件前缀 | 业务含义 |
| --- | --- |
| `F01_Sample_Clustering_Dendrogram` | 样本层次聚类树，用于观察样本整体相似性。 |
| `F02_Transcriptome_PCA` | 转录组 PCA 散点图，按分组展示样本差异。 |
| `F03_Transcriptome_PCA_Subgroups` | 转录组 PCA 二级分组图。 |
| `F04_Transcriptome_PCA_Pairs` | 转录组多个主成分成对散点图。 |
| `F05_Transcriptome_PCA_Pairs_Subgroups` | 转录组多主成分二级分组成对散点图。 |
| `F06_Metabolome_PCA` | 代谢组 PCA 散点图。 |
| `F07_Metabolome_PCA_Subgroups` | 代谢组 PCA 二级分组图。 |
| `F08_Metabolome_PCA_Pairs` | 代谢组多个主成分成对散点图。 |
| `F09_Metabolome_PCA_Pairs_Subgroups` | 代谢组多主成分二级分组成对散点图。 |
| `F10_Association_Evidence_UpSet` | PCC、Spearman、MI、ElasticNet、XGBoost 证据交集 UpSet 图。 |
| `F11_Gene_Metabolite_Correlation_Bubble_Heatmap` | 高置信基因-代谢物相关气泡热图。 |
| `F12_Top_Gene_Metabolite_Correlation_Heatmaps` | top 基因和 top 代谢物相关热图。 |
| `F13_Top_Gene_Metabolite_Pairs` | top 基因-代谢物关联回归散点图。 |
| `F14_Top_Metabolite_Group1_Violin_Box` | top 代谢物在 group1 下的分布图。 |
| `F15_Module_Eigengene_Heatmap` | 模块 eigengene 热图，带样本分组注释。 |
| `F16_Module_Eigengene_Heatmap_Group2` | 按二级分组组织的模块 eigengene 热图。 |
| `F17_Module_Zscore_Line_Panels` | 模块 z-score 折线面板图。 |
| `F18_Module_Gene_Zscore_Line_Panels` | 模块内基因 z-score 折线面板图。 |
| `F19_Module_Eigengene_Ridge` | 模块 eigengene ridge 分布图。 |
| `F20_Module_Eigengene_Ridge_Group1` | group1 覆盖的模块 eigengene ridge 分布图。 |
| `F21_Module_Eigengene_Group1_Violin_Box` | 模块 eigengene 在 group1 下的 violin/box 图。 |
| `F22_Module_kME_Boxplot` | 模块内 kME 分布图。 |
| `F23_Module_Metabolite_Association_Heatmap` | 模块-代谢物 Spearman 相关热图，带显著性星号。 |
| `F24_Module_Metabolite_Bubble_Plot` | 模块-代谢物相关气泡图。 |
| `F25_Module_Top_Metabolite_Regressions` | 模块 eigengene 与最相关代谢物的回归图。 |
| `F26_Module_Eigengene_Metabolite_Trend_Panels` | 模块 eigengene 与 top 代谢物趋势对照图。 |
| `F27_Association_Direction_Summary` | 模块内正负关联方向统计图。 |
| `F28_EdgeWeight_Distribution_By_Module` | 模块内高置信边权重分布图。 |
| `F29_Compressed_Circos_Network` | 压缩版 Circos 关联网络图。 |
| `F30_Floating_CNet_Circos_Network` | cnetplot 风格的圆形关联网络图。 |

## 12. 当前交互式可视化能力

`OmicsPrism_Interactive_Report.html` 是单文件浏览器应用。它的特点是：

- 不依赖后端服务。
- 数据 payload、schema、初始状态和前端渲染逻辑都内嵌在 HTML 中。
- 使用浏览器原生 SVG 绘图。
- 支持交互筛选、参数调整、hover/click 检查和 SVG 导出。

当前交互视图包括：

| 视图 ID | 标题 | 作用 |
| --- | --- | --- |
| `pca` | PCA Explorer | 切换转录组/代谢组 PCA，选择主成分、着色方式、点大小、标签等。 |
| `association` | Association Scatter Studio | 查看基因-代谢物或模块-代谢物回归散点图。 |
| `module_heatmap` | Module Heatmap Studio | 交互式模块-代谢物相关热图。 |
| `network_explorer` | Network Explorer | 高置信基因-代谢物二部网络浏览，支持节点 hover 和邻居高亮。 |

交互报告的后端构造思想：

- 为每类视图构建 dataset payload。
- 为每类视图构建 controls schema。
- 为报告构建 initial state。
- 前端根据 active view、schema 和 controls 渲染 SVG。

## 13. 关键配置对象

主要配置类是 `AnalysisConfig`。常用字段如下：

| 配置 | 默认值 | 说明 |
| --- | --- | --- |
| `project_name` | `OmicsPrism_Association_Analysis` | 项目名。 |
| `output_dir` | `results` | 输出目录。 |
| `group_table_path` | 必填 | 分组表路径。 |
| `random_state` | 42 | 随机种子。 |
| `missing_feature_threshold` | 0.5 | 缺失比例过滤阈值。 |
| `knn_neighbors` | 5 | KNN 填补邻居数。 |
| `trans_log2` | True | 是否对转录组做 log2 转换。 |
| `screen_top_k_per_method` | 1000 | 每种筛选方法保留的候选基因数。 |
| `fdr_alpha` | 0.05 | FDR 阈值。 |
| `selection_ratio` | 0.20 | 每个代谢物最终目标特征数比例。 |
| `min_features` | 10 | 每个代谢物至少选择的特征数。 |
| `max_features` | 50 | 每个代谢物最多选择的特征数。 |
| `network_plot_top_edges` | 120 | 网络图默认展示边数。 |
| `top_pairs_plot_n` | 6 | top 回归图默认展示数量。 |
| `enable_module_detection` | True | 是否启用模块检测。 |
| `module_method` | `leiden` | 模块检测方法。 |
| `generate_reports` | True | 是否生成报告。 |
| `export_pdf/svg/png` | True | 是否导出对应图格式。 |
| `export_cytoscape` | True | 是否导出 Cytoscape 表。 |
| `save_h5ad` | True | 是否保存 H5AD 状态。 |

## 14. 项目技术栈

主要依赖：

- Python 3.9+
- pandas / numpy：表格和矩阵计算。
- scipy / scikit-learn：统计、PCA、KNNImputer、标准化、ElasticNet 等。
- xgboost：非线性关联建模。
- anndata：多组学数据容器。
- matplotlib / seaborn：静态图。
- python-igraph / leidenalg：Leiden 模块检测。
- click：命令行接口。

当前前端交互报告没有使用 React/Vue 等框架，而是生成单个 HTML 文件，并在其中使用原生 JavaScript 和 SVG。
