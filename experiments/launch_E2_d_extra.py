# NOTE: service address and credentials are read from environment
#       variables (LLM_BASE_URL / LLM_API_KEY / LLM_MODEL).
#       This file was sanitised before public release.
# -*- coding: utf-8 -*-
"""方案 A2 — E2 min_depth 敏感性扫描【扩样补充实验】

与 launch_E1_K_extra.py 同思路：复用 select_balanced_samples 选 50 题，取后 25 题为子集。
固定 K=10，扫描 d ∈ {1, 2, 3}。
"""
import os, sys, json, shutil, subprocess, time
from datetime import datetime

POG_DIR = "/opt/notebook/workspace/pog_exp"
sys.path.insert(0, POG_DIR)
from run_experiment import select_balanced_samples

D_VALUES         = [1, 2, 3]
ORIG_SAMPLE_SIZE = 25
TOTAL_TARGET     = 50
EXTRA_SIZE       = TOTAL_TARGET - ORIG_SAMPLE_SIZE
QUESTION_TIMEOUT = 1500
SRC_DATASET      = os.path.join(POG_DIR, "experiment_questions.json")
NT_FILE          = "/opt/notebook/workspace/military_experiment/military_kg_group1.nt"

OUTPUT_ROOT = os.path.join(POG_DIR, "results", f"E2_d_extra_{datetime.now():%Y%m%d_%H%M%S}")
os.makedirs(OUTPUT_ROOT, exist_ok=True)

def build_extra_subset():
    with open(SRC_DATASET, "r", encoding="utf-8") as f:
        all_data = json.load(f)
    valid = [d for d in all_data if d.get("topic_entity")]
    selected_50 = select_balanced_samples(valid, TOTAL_TARGET)
    extra_25 = selected_50[ORIG_SAMPLE_SIZE:]
    extra_path = os.path.join(OUTPUT_ROOT, "experiment_questions_extra25.json")
    with open(extra_path, "w", encoding="utf-8") as f:
        json.dump(extra_25, f, ensure_ascii=False, indent=2)
    print(f"[A2] 写入 {extra_path}（{len(extra_25)} 道）")
    return extra_path

EXTRA_DATASET = build_extra_subset()

COMMON_ENV = {
    "LLM_BASE_URL": "os.environ.get('LLM_BASE_URL', 'http://localhost:11434/v1')",
    "LLM_API_KEY":  "ollama", "LLM_MODEL":    "qwen3:32b",
    "NT_FILE":      NT_FILE,
    "POG_DATASET_FILE":  EXTRA_DATASET,
    "POG_SAMPLE_SIZE":   str(EXTRA_SIZE),
    "POG_SAMPLE_STRATEGY": "ordered",
    "POG_MAX_DEPTH":     "3", "POG_FAST_MODE":     "0",
    "POG_PAPER_QA_CHAIN": "1", "POG_USE_LEGACY_KG_GATE": "0",
    "POG_RELATION_FILTER_THRESHOLD": "15",
    "POG_RELATION_TOPK": "10",   # E2 中 K 固定为 10
    "POG_ACTIVE_CONFIG_NAMES": '["D_双模块全开"]',
    "POG_QUESTION_TIMEOUT": str(QUESTION_TIMEOUT),
}

RESULTS_DIR_GLOBAL = os.path.join(POG_DIR, "results")
def _snapshot():
    if not os.path.exists(RESULTS_DIR_GLOBAL): return set()
    return {fn for fn in os.listdir(RESULTS_DIR_GLOBAL) if fn.startswith("pog_4group_")}

def run_one_d(d):
    out_dir = os.path.join(OUTPUT_ROOT, f"d_{d}")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "console.log")
    before = _snapshot()
    env = os.environ.copy(); env.update(COMMON_ENV)
    env["POG_MIN_DEPTH_BEFORE_STOP"] = str(d)
    env["POG_CHECKPOINT_PATH"] = os.path.join(out_dir, "resume_checkpoint.json")
    print(f"\n{'='*70}\n[A2·E2] d = {d}  开始：{datetime.now():%H:%M:%S}\n{'='*70}\n")
    cmd = [sys.executable, os.path.join(POG_DIR, "run_experiment.py")]
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.Popen(cmd, cwd=POG_DIR, env=env, stdout=logf, stderr=subprocess.STDOUT)
        proc.wait()
    print(f"[A2·E2] d = {d}  完成（exit={proc.returncode}）")
    for fn in sorted(_snapshot() - before):
        try:
            shutil.move(os.path.join(RESULTS_DIR_GLOBAL, fn), os.path.join(out_dir, fn))
            print(f"  [迁移] {fn} -> {out_dir}")
        except Exception as e:
            print(f"  [WARN] {e}")
    return out_dir

if __name__ == "__main__":
    if not os.path.exists(NT_FILE): print(f"[FATAL] {NT_FILE}"); sys.exit(1)
    print(f"[A2 启动] 输出根目录: {OUTPUT_ROOT}")
    print(f"[A2 启动] 扩样集大小 {EXTRA_SIZE}，扫描 d = {D_VALUES}")
    t0 = time.time()
    for d in D_VALUES: run_one_d(d)
    print(f"\n[A2 完成] 总耗时 {(time.time()-t0)/3600:.2f} h")
