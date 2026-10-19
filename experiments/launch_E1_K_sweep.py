# NOTE: service address and credentials are read from environment
#       variables (LLM_BASE_URL / LLM_API_KEY / LLM_MODEL).
#       This file was sanitised before public release.
# -*- coding: utf-8 -*-
"""E1 — 关系筛选 K 值敏感性扫描

设计：
  - 固定 D 组（双模块全开）+ G1 批次 + balanced 采样 25 题
  - 扫描 K ∈ {5, 10, 15, 20}
  - 每个 K 独立目录输出 JSON / MD / 日志

用法（远端 pog_exp 目录下）：
  cp launch_E1_K_sweep.py /opt/notebook/workspace/pog_exp/
  python /opt/notebook/workspace/pog_exp/launch_E1_K_sweep.py

继承 run_experiment.py 全部基础设施（数据加载、单题超时、断点续跑、统计）。
"""
import os, sys, json, shutil, signal, subprocess, time
from datetime import datetime

# ============================================================================
# 实验配置
# ============================================================================
K_VALUES         = [5, 10, 15, 20]
SAMPLE_SIZE      = 25
SAMPLE_STRATEGY  = "balanced_by_compositionality"
QUESTION_TIMEOUT = 1500
DATASET_FILE     = "experiment_questions.json"  # G1 数据集（远端实际文件名）
NT_FILE          = "/opt/notebook/workspace/military_experiment/military_kg_group1.nt"

POG_DIR          = "/opt/notebook/workspace/pog_exp"
OUTPUT_ROOT      = os.path.join(POG_DIR, "results", f"E1_K_sweep_{datetime.now():%Y%m%d_%H%M%S}")
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ============================================================================
# 公共环境（D 组配置 + 模型）
# ============================================================================
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
    "POG_RELATION_FILTER_THRESHOLD": "15",   # 触发门槛固定
    "POG_MIN_DEPTH_BEFORE_STOP":     "2",    # min_depth 在 E1 中固定
    "POG_ACTIVE_CONFIG_NAMES": '["D_双模块全开"]',  # 仅 D 组
    "POG_QUESTION_TIMEOUT": str(QUESTION_TIMEOUT),
}

# ============================================================================
# 主循环
# ============================================================================
RESULTS_DIR_GLOBAL = os.path.join(POG_DIR, "results")

def _snapshot_results():
    if not os.path.exists(RESULTS_DIR_GLOBAL): return set()
    return {fn for fn in os.listdir(RESULTS_DIR_GLOBAL)
            if fn.startswith("pog_4group_")}

def run_one_K(K):
    out_dir = os.path.join(OUTPUT_ROOT, f"K_{K}")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "console.log")

    before = _snapshot_results()

    env = os.environ.copy()
    env.update(COMMON_ENV)
    env["POG_RELATION_TOPK"] = str(K)
    # 隔离断点缓存到本 K 的目录（CHECKPOINT_PATH 是 run_experiment 实际识别的 env var）
    env["POG_CHECKPOINT_PATH"] = os.path.join(out_dir, "resume_checkpoint.json")

    print(f"\n{'='*70}\n[E1] K = {K}  开始：{datetime.now():%H:%M:%S}\n{'='*70}\n")
    cmd = [sys.executable, os.path.join(POG_DIR, "run_experiment.py")]
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.Popen(cmd, cwd=POG_DIR, env=env,
                                stdout=logf, stderr=subprocess.STDOUT)
        proc.wait()
    print(f"[E1] K = {K}  完成（exit={proc.returncode}）  日志: {log_path}")

    # run_experiment 把 pog_4group_<时间戳>.json/.md 写到默认 results/ ；移动到本 K 子目录
    after = _snapshot_results()
    new_files = sorted(after - before)
    for fn in new_files:
        src = os.path.join(RESULTS_DIR_GLOBAL, fn)
        dst = os.path.join(out_dir, fn)
        try:
            shutil.move(src, dst); print(f"  [迁移] {fn} -> {out_dir}")
        except Exception as e:
            print(f"  [WARN] 迁移失败 {fn}: {e}")
    return out_dir

if __name__ == "__main__":
    if not os.path.exists(NT_FILE):
        print(f"[FATAL] KG 文件不存在: {NT_FILE}"); sys.exit(1)

    print(f"[E1 启动] 输出根目录: {OUTPUT_ROOT}")
    print(f"[E1 启动] 扫描 K = {K_VALUES}")
    print(f"[E1 启动] 每 K 跑 {SAMPLE_SIZE} 题，预计单 K ~{SAMPLE_SIZE*350/3600:.1f} h")
    t0 = time.time()
    for K in K_VALUES:
        run_one_K(K)
    elapsed_h = (time.time() - t0) / 3600
    print(f"\n[E1 完成] 总耗时 {elapsed_h:.2f} h；输出根目录: {OUTPUT_ROOT}")
