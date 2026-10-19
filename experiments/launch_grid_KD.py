# NOTE: service address and credentials are read from environment
#       variables (LLM_BASE_URL / LLM_API_KEY / LLM_MODEL).
#       This file was sanitised before public release.
# -*- coding: utf-8 -*-
"""方案 D — K × min_depth 联合栅格扫描

固定 G1 + balanced 25 题（与首次 E1/E2 同样本，便于配对对比）。
扫描三个对角点：
  - (K=5,  d=1)  → 极激进配置
  - (K=10, d=2)  → 论文默认配置（与首次 E1 K=10 / 首次 E2 d=2 已有数据，本组重跑作为基准 sanity check）
  - (K=20, d=3)  → 极保守配置

目的：验证两改进模块在不同强度下是否独立作用、是否存在交互效应。
"""
import os, sys, shutil, subprocess, time
from datetime import datetime

GRID = [
    ("K5_d1",   5, 1),
    ("K10_d2", 10, 2),
    ("K20_d3", 20, 3),
]
SAMPLE_SIZE      = 25
SAMPLE_STRATEGY  = "balanced_by_compositionality"
QUESTION_TIMEOUT = 1500
DATASET_FILE     = "experiment_questions.json"
NT_FILE          = "/opt/notebook/workspace/military_experiment/military_kg_group1.nt"

POG_DIR     = "/opt/notebook/workspace/pog_exp"
OUTPUT_ROOT = os.path.join(POG_DIR, "results", f"D_grid_KD_{datetime.now():%Y%m%d_%H%M%S}")
os.makedirs(OUTPUT_ROOT, exist_ok=True)

COMMON_ENV = {
    "LLM_BASE_URL": "os.environ.get('LLM_BASE_URL', 'http://localhost:11434/v1')",
    "LLM_API_KEY":  "ollama", "LLM_MODEL":    "qwen3:32b",
    "NT_FILE":      NT_FILE,
    "POG_DATASET_FILE": os.path.join(POG_DIR, DATASET_FILE),
    "POG_SAMPLE_SIZE":  str(SAMPLE_SIZE),
    "POG_SAMPLE_STRATEGY": SAMPLE_STRATEGY,
    "POG_MAX_DEPTH":    "3", "POG_FAST_MODE": "0",
    "POG_PAPER_QA_CHAIN": "1", "POG_USE_LEGACY_KG_GATE": "0",
    "POG_RELATION_FILTER_THRESHOLD": "15",
    "POG_ACTIVE_CONFIG_NAMES": '["D_双模块全开"]',
    "POG_QUESTION_TIMEOUT": str(QUESTION_TIMEOUT),
}

RESULTS_DIR_GLOBAL = os.path.join(POG_DIR, "results")
def _snapshot():
    if not os.path.exists(RESULTS_DIR_GLOBAL): return set()
    return {fn for fn in os.listdir(RESULTS_DIR_GLOBAL) if fn.startswith("pog_4group_")}

def run_one(name, K, d):
    out_dir = os.path.join(OUTPUT_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "console.log")
    before = _snapshot()
    env = os.environ.copy(); env.update(COMMON_ENV)
    env["POG_RELATION_TOPK"] = str(K)
    env["POG_MIN_DEPTH_BEFORE_STOP"] = str(d)
    env["POG_CHECKPOINT_PATH"] = os.path.join(out_dir, "resume_checkpoint.json")
    print(f"\n{'='*70}\n[D·栅格] {name} (K={K}, d={d})  开始：{datetime.now():%H:%M:%S}\n{'='*70}\n")
    cmd = [sys.executable, os.path.join(POG_DIR, "run_experiment.py")]
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.Popen(cmd, cwd=POG_DIR, env=env, stdout=logf, stderr=subprocess.STDOUT)
        proc.wait()
    print(f"[D·栅格] {name}  完成（exit={proc.returncode}）")
    for fn in sorted(_snapshot() - before):
        try:
            shutil.move(os.path.join(RESULTS_DIR_GLOBAL, fn), os.path.join(out_dir, fn))
            print(f"  [迁移] {fn} -> {out_dir}")
        except Exception as e:
            print(f"  [WARN] {e}")
    return out_dir

if __name__ == "__main__":
    if not os.path.exists(NT_FILE): print(f"[FATAL] {NT_FILE}"); sys.exit(1)
    print(f"[D 启动] 输出根目录: {OUTPUT_ROOT}")
    print(f"[D 启动] G1 + balanced{SAMPLE_SIZE} 题，对角扫描 = {[(n,K,d) for n,K,d in GRID]}")
    t0 = time.time()
    for name, K, d in GRID: run_one(name, K, d)
    print(f"\n[D 完成] 总耗时 {(time.time()-t0)/3600:.2f} h")
