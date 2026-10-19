# NOTE: service address and credentials are read from environment
#       variables (LLM_BASE_URL / LLM_API_KEY / LLM_MODEL).
#       This file was sanitised before public release.
# -*- coding: utf-8 -*-
"""
全局配置文件
所有参数在此集中管理，修改此文件即可调整实验行为

架构说明（与两篇核心文献一致，全文见 docs/extracted_texts/）：
- 主干：Tan 等 PoG（WWW 2025）「Paths-over-Graph: Knowledge Graph Empowered Large
  Language Model Reasoning」— 三阶段路径探索、多步剪枝（含 SBERT/LLM）、附录 E
  问答链（POG_PAPER_QA_CHAIN）；DOI 10.1145/3696410.3714892。
- 两开关：蒋献等 LLMKG（《电信科学》2025，doi:10.11959/j.issn.1000-0801.2025238）
  「基于大语言模型与知识图谱融合的多跳问答技术研究」— Prompt 注入与关系筛选、
  知识是否充足以决定是否继续检索；映射为 enable_relation_filter /
  enable_knowledge_judge，不替换 PoG 主干。
"""

import json
import os

# ================================================================
# 路径配置
# ================================================================
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
SYSTEM_DIR   = os.path.join(BASE_DIR, "system")
RESULTS_DIR  = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ================================================================
# 数据集配置
# ================================================================
# 数据集文件名/路径。
# 支持三种方式：
# 1) POG_DATASET_FILE 直接给绝对路径
# 2) POG_DATASET_FILE 给文件名，则按项目目录、上级目录、当前工作目录自动查找
# 3) 不设置时默认使用官方 CWQ test 集（ComplexWebQuestions_test.json）
DATASET_FILE = os.getenv("POG_DATASET_FILE", "ComplexWebQuestions_test.json")
NT_FILE = "/opt/notebook/workspace/pog_exp/military_kg_group2.nt"

def _resolve_path(path_value, fallback_name=None):
    """在本地与平台环境下自动解析可用路径。"""
    if not path_value:
        path_value = ""
    path_value = str(path_value).strip()
    if path_value and os.path.isabs(path_value) and os.path.exists(path_value):
        return path_value

    candidates = []
    if path_value:
        candidates.append(path_value)
    if fallback_name and fallback_name not in candidates:
        candidates.append(fallback_name)

    search_roots = [BASE_DIR, os.path.dirname(BASE_DIR), os.getcwd()]
    for root in search_roots:
        for cand in candidates:
            full = cand if os.path.isabs(cand) else os.path.join(root, cand)
            if os.path.exists(full):
                return full
    return os.path.join(BASE_DIR, path_value or (fallback_name or ""))


DATASET_PATH = _resolve_path(DATASET_FILE, "ComplexWebQuestions_test.json")

# 采样数量（设为 0 则使用全量数据）
SAMPLE_SIZE = int(os.getenv("POG_SAMPLE_SIZE", "200"))

# 采样策略："balanced_by_compositionality" 或 "sequential"
SAMPLE_STRATEGY = os.getenv("POG_SAMPLE_STRATEGY", "balanced_by_compositionality")

# ================================================================
# 知识图谱配置
# ================================================================
# 本地 .nt 文件路径（留空则自动查找）
# 也可通过环境变量 NT_FILE 指定
# # NT_FILE = _resolve_path(os.getenv("NT_FILE", ""), "cwq_topic_subgraph_targetA.nt")

# 远程 SPARQL 端点（留空则使用本地 rdflib）
# 也可通过环境变量 SPARQL_ENDPOINT 指定
SPARQL_ENDPOINT = os.getenv("SPARQL_ENDPOINT", "")

# ================================================================
# LLM 配置（通过环境变量注入，代码不硬编码密钥）
# ================================================================
# 使用前请设置以下环境变量：
#   set LLM_BASE_URL=http://your-api-host/v1
#   set LLM_API_KEY=your-api-key
#   set LLM_MODEL=your-model-name
#
# 示例（ollama本地）：
#   set LLM_BASE_URL=http://localhost:11434/v1
#   set LLM_API_KEY=ollama
#   set LLM_MODEL=qwen3:32b

# ================================================================
# PoG 算法参数
# ================================================================
MAX_DEPTH = int((os.getenv("POG_MAX_DEPTH", "3") or "3").strip() or "3")  # 最大跳数

# 是否要求所有初始实体连通才算成功
EVAL_REQUIRE_CONNECT_FOR_SUCCESS = False

# 两改进模块共用超参（蒋献等 LLMKG 式设定在工程上的对应；PoG 主干仍用上文 POG_*）
# 小样本粗扫推荐：Top-K=12，min/max 证据=8/50（见论文 §6.4.1）；可用环境变量覆盖。
PAPER_RELATION_TOPK       = int((os.getenv("POG_RELATION_TOPK", "10") or "10").strip() or "10")    # 每跳保留的最大关系数
PAPER_RELATION_FILTER_THRESHOLD = int((os.getenv("POG_RELATION_FILTER_THRESHOLD", "15") or "15").strip() or "15")  # 关系数≤此值时跳过筛选
PAPER_MIN_EVIDENCE        = int((os.getenv("POG_MIN_EVIDENCE", "8") or "8").strip() or "8")    # 判断为"充足"所需最少证据条数
PAPER_MAX_EVIDENCE        = int((os.getenv("POG_MAX_EVIDENCE", "50") or "50").strip() or "50")   # 超过此数直接判断为"充足"
PAPER_MIN_REL_TYPES       = int((os.getenv("POG_MIN_REL_TYPES", "4") or "4").strip() or "4")    # 充足所需最少关系类型数
PAPER_MIN_DEPTH_BEFORE_STOP = int((os.getenv("POG_MIN_DEPTH_BEFORE_STOP", "2") or "2").strip() or "2")  # 最少探索到第几跳才允许早停


# ================================================================
# PoG 论文默认搜索/剪枝宽度（对齐 WWW'25 PoG 设置）
# ================================================================
# W1: fuzzy selection 的候选上限（论文中默认 W1=80）
POG_W1_FUZZY = int(os.getenv("POG_W1_FUZZY", "80").strip() or "80")
# Wmax: 最终保留的 Top reasoning paths 宽度（论文中默认 Wmax=3）
POG_WMAX = int(os.getenv("POG_WMAX", "3").strip() or "3")

# PoG 论文 4.4：Path Summarizing → Question Answering Evaluation → Generation
# 设为 0 可退回「单条 prompt 直接作答」（旧行为）
POG_PAPER_QA_CHAIN = os.getenv("POG_PAPER_QA_CHAIN", "1").strip() == "1"
# 1=沿用旧版「关系<50 且实体<30 则拒答」门槛；0=对齐论文：由 LLM 结合路径与自身知识作答（仅无路径且无三元组时可仍走生成）
POG_USE_LEGACY_KG_GATE = os.getenv("POG_USE_LEGACY_KG_GATE", "0").strip() == "1"

# ================================================================
# 实验组配置（四组对比：A=完整PoG基线；B/C/D=在A上开关两个改进模块）
# ================================================================
EXPERIMENT_CONFIGS = [
    {
        "name":      "A_原始PoG",
        "desc":      "完整 PoG 流程（论文一致）；两改进模块均关闭",
        "filter":    False,
        "judge":     False,
        "llm_filter": False,
        "llm_judge": False,
    },
    {
        "name":      "B_仅关系筛选",
        "desc":      "PoG + 仅启用关系筛选 Top-K",
        "filter":    True,
        "judge":     False,
        "llm_filter": True,
        "llm_judge": False,
    },
    {
        "name":      "C_仅知识判断",
        "desc":      "PoG + 仅启用知识充足性判断（动态早停）",
        "filter":    False,
        "judge":     True,
        "llm_filter": False,
        "llm_judge": True,
    },
    {
        "name":      "D_双模块全开",
        "desc":      "PoG + 关系筛选与知识判断同时启用",
        "filter":    True,
        "judge":     True,
        "llm_filter": True,
        "llm_judge": True,
    },
    # [D-PRIME] 诊断组：判断照常调用并写日志，但不因"充足"而提前停止
    {
        "name":      "D_prime_判断不停",
        "desc":      "D' 诊断组：保留充足性判断调用与日志，忽略停止结果，探索至最大深度",
        "filter":    True,
        "judge":     True,
        "llm_filter": True,
        "llm_judge": True,
        "no_stop":   True,
    },
]

# 留空 [] 表示运行全部组；也可设环境变量 POG_ACTIVE_CONFIG_NAMES（JSON 数组）
# 示例：set POG_ACTIVE_CONFIG_NAMES=["A_原始PoG"]
_active_raw = os.getenv("POG_ACTIVE_CONFIG_NAMES", "").strip()
if _active_raw:
    try:
        ACTIVE_CONFIG_NAMES = json.loads(_active_raw)
    except json.JSONDecodeError:
        ACTIVE_CONFIG_NAMES = []
else:
    ACTIVE_CONFIG_NAMES = []

# ================================================================
# 快速模式（仅调试用：无 LLM API 时）
# ================================================================
# 默认 False：与开题/PoG/LLMKG 一致，走 LLM（关系筛选、知识判断、路径剪枝、附录 E 问答链等）。
# 本地无大模型或只做连通性测试时：set POG_FAST_MODE=1
FAST_MODE = os.getenv("POG_FAST_MODE", "0").strip() == "1"

