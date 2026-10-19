# NOTE: service address and credentials are read from environment
#       variables (LLM_BASE_URL / LLM_API_KEY / LLM_MODEL).
#       This file was sanitised before public release.
# -*- coding: utf-8 -*-
"""
PoG 四组对比实验 - 平台启动器
自动配置环境变量并运行实验，输出详细中文日志
"""
import os
import sys
import time
from datetime import datetime

# ====== 检查依赖 ======
try:
    import sentence_transformers
    print(f"[依赖] sentence-transformers v{sentence_transformers.__version__} - SBERT可用")
except ImportError:
    print("[依赖] sentence-transformers 未安装，Fuzzy Selection将使用关键词匹配兜底（不影响实验）")
print()

# ====== 环境配置 ======
os.environ["LLM_BASE_URL"] = "os.environ.get('LLM_BASE_URL', 'http://localhost:11434/v1')"
os.environ["LLM_API_KEY"] = "ollama"
os.environ["LLM_MODEL"] = "qwen3:32b"
os.environ["NT_FILE"] = "/opt/notebook/workspace/military_experiment/military_kg_group1.nt"
os.environ["POG_DATASET_FILE"] = "/opt/notebook/workspace/pog_exp/experiment_questions.json"
os.environ["POG_SAMPLE_SIZE"] = "100"  # 第1组全部100题
os.environ["POG_SAMPLE_STRATEGY"] = "balanced_by_compositionality"
os.environ["POG_MAX_DEPTH"] = "3"
os.environ["POG_FAST_MODE"] = "0"  # 使用LLM
os.environ["POG_PAPER_QA_CHAIN"] = "1"
os.environ["POG_USE_LEGACY_KG_GATE"] = "0"
os.environ["POG_RELATION_TOPK"] = "10"
os.environ["POG_RELATION_FILTER_THRESHOLD"] = "15"
os.environ["POG_MIN_DEPTH_BEFORE_STOP"] = "2"
os.environ["POG_ACTIVE_CONFIG_NAMES"] = '["A_原始PoG", "B_仅关系筛选", "C_仅知识判断", "D_双模块全开"]'

# 切换工作目录
os.chdir("/opt/notebook/workspace/pog_exp")
sys.path.insert(0, "/opt/notebook/workspace/pog_exp")

# ====== 日志重定向 ======
LOG_FILE = f"/opt/notebook/workspace/pog_exp/results/experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

class TeeWriter:
    """同时输出到控制台和文件"""
    def __init__(self, filepath):
        self.file = open(filepath, "w", encoding="utf-8")
        self.stdout = sys.stdout
    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
        self.file.flush()
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    def close(self):
        self.file.close()

tee = TeeWriter(LOG_FILE)
sys.stdout = tee
sys.stderr = tee

print("=" * 70)
print(f"PoG 四组对比实验 - 详细中文日志")
print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"日志文件: {LOG_FILE}")
print("=" * 70)
print()
print(f"【环境配置】")
print(f"  LLM接口: {os.environ['LLM_BASE_URL']}")
print(f"  LLM模型: {os.environ['LLM_MODEL']}")
print(f"  知识图谱: {os.environ['NT_FILE']}")
print(f"  数据集: {os.environ['POG_DATASET_FILE']}")
print(f"  采样数: {os.environ['POG_SAMPLE_SIZE']} 题")
print(f"  最大跳数: {os.environ['POG_MAX_DEPTH']}")
print(f"  关系筛选K: {os.environ['POG_RELATION_TOPK']}")
print(f"  知识判断最小深度: {os.environ['POG_MIN_DEPTH_BEFORE_STOP']}")
print()

# ====== 检查LLM连通性 ======
print("【步骤1/4】检查LLM接口连通性...")
try:
    import requests as req
    test_url = os.environ["LLM_BASE_URL"].rstrip("/") + "/models"
    r = req.get(test_url, timeout=10)
    if r.status_code == 200:
        models = r.json().get("data", [])
        model_names = [m.get("id", "") for m in models]
        print(f"  [OK] LLM接口正常，可用模型: {model_names[:5]}")
    else:
        print(f"  [WARN] LLM接口返回 {r.status_code}")
except Exception as e:
    print(f"  [ERROR] LLM接口不可用: {e}")
    print("  将使用兜底规则模式运行")
    os.environ["POG_FAST_MODE"] = "1"
print()

# ====== 检查KG文件 ======
print("【步骤2/4】检查知识图谱文件...")
nt_path = os.environ["NT_FILE"]
if os.path.exists(nt_path):
    size_mb = os.path.getsize(nt_path) / 1024 / 1024
    print(f"  [OK] KG文件存在: {nt_path} ({size_mb:.1f} MB)")
else:
    print(f"  [ERROR] KG文件不存在: {nt_path}")
    # 尝试备选路径
    alt_paths = [
        "/opt/notebook/workspace/cc/cwq_military_subgraph.nt",
        "/opt/notebook/workspace/military_experiment/output.nt",
    ]
    for alt in alt_paths:
        if os.path.exists(alt):
            print(f"  [INFO] 使用备选: {alt}")
            os.environ["NT_FILE"] = alt
            break
print()

# ====== 检查数据集 ======
print("【步骤3/4】检查实验数据集...")
ds_path = os.environ["POG_DATASET_FILE"]
if os.path.exists(ds_path):
    import json
    with open(ds_path, "r", encoding="utf-8") as f:
        ds = json.load(f)
    print(f"  [OK] 数据集加载成功: {len(ds)} 问题")
    # 统计类型分布
    types = {}
    for item in ds:
        t = item.get("compositionality_type", "unknown")
        types[t] = types.get(t, 0) + 1
    print(f"  问题类型分布: {types}")
else:
    print(f"  [ERROR] 数据集不存在: {ds_path}")
    sys.exit(1)
print()

# ====== 运行实验 ======
print("【步骤4/4】开始运行四组对比实验...")
print("=" * 70)
print()

try:
    # 导入并执行
    exec(open("/opt/notebook/workspace/pog_exp/run_experiment.py").read())
except Exception as e:
    import traceback
    print(f"\n[FATAL] 实验运行失败: {e}")
    traceback.print_exc()

print()
print("=" * 70)
print(f"实验结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"完整日志已保存: {LOG_FILE}")
print("=" * 70)

tee.close()
sys.stdout = tee.stdout
