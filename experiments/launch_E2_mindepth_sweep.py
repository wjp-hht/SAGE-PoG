# NOTE: service address and credentials are read from environment
#       variables (LLM_BASE_URL / LLM_API_KEY / LLM_MODEL).
#       This file was sanitised before public release.
# -*- coding: utf-8 -*-
"""E2 — 知识充足性判断的最小深度门槛 min_depth 扫描

设计：
  - 固定 D 组（双模块全开）+ G1 批次 + balanced 采样 25 题
  - 扫描 min_depth ∈ {1, 2, 3}  （d=2 是论文默认；d=1 更激进；d=3 更保守）
  - 每个取值独立目录输出
"""
import os, sys, subprocess, time
from datetime import datetime

MIN_DEPTH_VALUES = [1, 2, 3]
SAMPLE_SIZE      = 25
SAMPLE_STRATEGY  = "balanced_by_compositionality"
QUESTION_TIMEOUT = 1500
DATASET_FILE     = "experiment_questions.json"  # G1 数据集
NT_FILE          = "/opt/notebook/workspace/military_experiment/military_kg_group1.nt"
POG_DIR          = "/opt/notebook/workspace/pog_exp"
OUTPUT_ROOT      = os.path.join(POG_DIR, "results",
                                f"E2_mindepth_sweep_{datetime.now():%Y%m%d_%H%M%S}")
os.makedirs(OUTPUT_ROOT, exist_ok=True)

COMMON_ENV = {
    "LLM_BASE_URL": "os.environ.get('LLM_BASE_URL', 'http://localhost:11434/v1')",
    "LLM_API_KEY":  "ollama",
    "LLM_MODEL":    "qwen3:32b",
    "NT_FILE":      NT_FILE,
    "POG_DATASET_FILE": os.path.join(POG_DIR, DATASET_FILE),
    "POG_SAMPLE_SIZE":  str(SAMPLE_SIZE),
    "POG_SAMPLE_STRATEGY": SAMPLE_STRATEGY,
    "POG_MAX_DEPTH":    "3",
    "POG_FAST_MODE":    "0",
    "POG_PAPER_QA_CHAIN": "1",
    "POG_USE_LEGACY_KG_GATE": "0",
    "POG_RELATION_TOPK":             "10",   # K 在 E2 中固定为论文默认
    "POG_RELATION_FILTER_THRESHOLD": "15",
    "POG_ACTIVE_CONFIG_NAMES": '["D_双模块全开"]',
    "POG_QUESTION_TIMEOUT": str(QUESTION_TIMEOUT),
}

RESULTS_DIR_GLOBAL = os.path.join(POG_DIR, "results")

def _snapshot_results():
    if not os.path.exists(RESULTS_DIR_GLOBAL): return set()
    return {fn for fn in os.listdir(RESULTS_DIR_GLOBAL) if fn.startswith("pog_4group_")}

def run_one(d):
    import shutil
    out_dir = os.path.join(OUTPUT_ROOT, f"d_{d}")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "console.log")
    before = _snapshot_results()
    env = os.environ.copy()
    env.update(COMMON_ENV)
    env["POG_MIN_DEPTH_BEFORE_STOP"] = str(d)
    env["POG_CHECKPOINT_PATH"]       = os.path.join(out_dir, "resume_checkpoint.json")
    print(f"\n{'='*70}\n[E2] min_depth = {d}  开始：{datetime.now():%H:%M:%S}\n{'='*70}\n")
    cmd = [sys.executable, os.path.join(POG_DIR, "run_experiment.py")]
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.Popen(cmd, cwd=POG_DIR, env=env, stdout=logf, stderr=subprocess.STDOUT)
        proc.wait()
    print(f"[E2] min_depth = {d}  完成（exit={proc.returncode}）  日志: {log_path}")
    after = _snapshot_results()
    for fn in sorted(after - before):
        src = os.path.join(RESULTS_DIR_GLOBAL, fn); dst = os.path.join(out_dir, fn)
        try: shutil.move(src, dst); print(f"  [迁移] {fn} -> {out_dir}")
        except Exception as e: print(f"  [WARN] 迁移失败 {fn}: {e}")
    return out_dir

if __name__ == "__main__":
    if not os.path.exists(NT_FILE):
        print(f"[FATAL] KG 文件不存在: {NT_FILE}"); sys.exit(1)
    print(f"[E2 启动] 输出根目录: {OUTPUT_ROOT}")
    print(f"[E2 启动] 扫描 min_depth = {MIN_DEPTH_VALUES}")
    t0 = time.time()
    for d in MIN_DEPTH_VALUES:
        run_one(d)
    print(f"\n[E2 完成] 总耗时 {(time.time()-t0)/3600:.2f} h；输出根目录: {OUTPUT_ROOT}")
