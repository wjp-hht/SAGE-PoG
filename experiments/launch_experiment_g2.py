# NOTE: service address and credentials are read from environment
#       variables (LLM_BASE_URL / LLM_API_KEY / LLM_MODEL).
#       This file was sanitised before public release.
# -*- coding: utf-8 -*-
"""
PoG 四组对比实验 - G2 批次启动器
（由 upload_g2_setup.py 自动生成）
"""
import os
import sys
from datetime import datetime

# ====== 环境配置（指向 G2 数据集与 KG）======
os.environ["LLM_BASE_URL"] = "os.environ.get('LLM_BASE_URL', 'http://localhost:11434/v1')"
os.environ["LLM_API_KEY"] = "ollama"
os.environ["LLM_MODEL"] = "qwen3:32b"
os.environ["NT_FILE"] = "/opt/notebook/workspace/military_experiment/military_kg_group2.nt"
os.environ["POG_DATASET_FILE"] = "/opt/notebook/workspace/pog_exp/experiment_questions_g2.json"
os.environ["POG_SAMPLE_SIZE"] = "100"
os.environ["POG_SAMPLE_STRATEGY"] = "balanced_by_compositionality"
os.environ["POG_MAX_DEPTH"] = "3"
os.environ["POG_FAST_MODE"] = "0"
os.environ["POG_PAPER_QA_CHAIN"] = "1"
os.environ["POG_USE_LEGACY_KG_GATE"] = "0"
os.environ["POG_RELATION_TOPK"] = "10"
os.environ["POG_RELATION_FILTER_THRESHOLD"] = "15"
os.environ["POG_MIN_DEPTH_BEFORE_STOP"] = "2"
os.environ["POG_ACTIVE_CONFIG_NAMES"] = '["A_原始PoG", "B_仅关系筛选", "C_仅知识判断", "D_双模块全开"]'

os.chdir("/opt/notebook/workspace/pog_exp")
sys.path.insert(0, "/opt/notebook/workspace/pog_exp")

LOG_FILE = f"/opt/notebook/workspace/pog_exp/results/experiment_g2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

class TeeWriter:
    def __init__(self, fp):
        self.file = open(fp, "w", encoding="utf-8")
        self.stdout = sys.stdout
    def write(self, d):
        self.stdout.write(d); self.file.write(d); self.file.flush()
    def flush(self):
        self.stdout.flush(); self.file.flush()
    def close(self):
        self.file.close()

tee = TeeWriter(LOG_FILE)
sys.stdout = tee
sys.stderr = tee

print("=" * 70)
print("PoG G2 批次 - 四组对比实验")
print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"日志: {LOG_FILE}")
print(f"KG:  {os.environ['NT_FILE']}")
print(f"数据集: {os.environ['POG_DATASET_FILE']}")
print("=" * 70)
print()

# ====== 前置检查 ======
nt_path = os.environ["NT_FILE"]
ds_path = os.environ["POG_DATASET_FILE"]
if not os.path.exists(nt_path):
    print(f"[FATAL] G2 KG 文件不存在: {nt_path}")
    print("请先执行 KG 抽取，等 military_kg_group2.nt 生成后再启动实验。")
    sys.exit(1)
if not os.path.exists(ds_path):
    print(f"[FATAL] 数据集不存在: {ds_path}")
    sys.exit(1)

import json as _json
with open(ds_path, "r", encoding="utf-8") as f:
    _ds = _json.load(f)
print(f"[OK] 数据集 {len(_ds)} 题，KG {os.path.getsize(nt_path)/1024/1024:.1f} MB")
print()

# ====== 运行实验 ======
try:
    exec(open("/opt/notebook/workspace/pog_exp/run_experiment.py").read())
except Exception as e:
    import traceback
    print(f"\n[FATAL] {e}")
    traceback.print_exc()

print()
print("=" * 70)
print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"日志: {LOG_FILE}")
print("=" * 70)
tee.close()
sys.stdout = tee.stdout
