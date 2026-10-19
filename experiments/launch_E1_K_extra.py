# NOTE: service address and credentials are read from environment
#       variables (LLM_BASE_URL / LLM_API_KEY / LLM_MODEL).
#       This file was sanitised before public release.
# -*- coding: utf-8 -*-
"""方案 A1 — E1 K 值敏感性扫描【扩样补充实验】

设计思路：
  - 与首次 E1（n=25）使用相同 balanced_by_compositionality 采样
  - 通过预生成"第 26-50 题"的子集临时文件，避免重跑前 25 题
  - 与首次结果合并后即可得到 n=50 的扩样数据

操作：
  1) 读取 G1 全量 experiment_questions.json
  2) 复用 select_balanced_samples 选出 50 题
  3) 取后 25 题（[25:50]）写到临时数据集文件
  4) 用临时数据集 + SAMPLE_STRATEGY=ordered + SAMPLE_SIZE=25 跑 K=5/10/15/20
"""
import os, sys, json, shutil, subprocess, time
from datetime import datetime

POG_DIR = "/opt/notebook/workspace/pog_exp"
sys.path.insert(0, POG_DIR)

# 复用 run_experiment 的均衡采样函数
from run_experiment import select_balanced_samples

# ==========================================================================
# 实验配置
# ==========================================================================
K_VALUES         = [5, 10, 15, 20]
ORIG_SAMPLE_SIZE = 25      # 首次 E1 用的样本量
TOTAL_TARGET     = 50      # 扩样目标
EXTRA_SIZE       = TOTAL_TARGET - ORIG_SAMPLE_SIZE   # 增量 = 25
QUESTION_TIMEOUT = 1500
SRC_DATASET      = os.path.join(POG_DIR, "experiment_questions.json")
NT_FILE          = "/opt/notebook/workspace/military_experiment/military_kg_group1.nt"

OUTPUT_ROOT = os.path.join(POG_DIR, "results", f"E1_K_extra_{datetime.now():%Y%m%d_%H%M%S}")
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ==========================================================================
# 生成"第 26-50 题"子集
# ==========================================================================
def build_extra_subset():
    print(f"[A1] 读取 {SRC_DATASET}")
    with open(SRC_DATASET, "r", encoding="utf-8") as f:
        all_data = json.load(f)
    valid = [d for d in all_data if d.get("topic_entity")]
    print(f"[A1] 全量 {len(all_data)}  有效 {len(valid)}")

    selected_50 = select_balanced_samples(valid, TOTAL_TARGET)
    extra_25 = selected_50[ORIG_SAMPLE_SIZE:]
    print(f"[A1] 选取扩样子集: 第 {ORIG_SAMPLE_SIZE+1}-{TOTAL_TARGET} 题，共 {len(extra_25)} 道")

    extra_path = os.path.join(OUTPUT_ROOT, "experiment_questions_extra25.json")
    with open(extra_path, "w", encoding="utf-8") as f:
        json.dump(extra_25, f, ensure_ascii=False, indent=2)
    print(f"[A1] 写入 {extra_path}")
    return extra_path

EXTRA_DATASET = build_extra_subset()

# ==========================================================================
# 公共环境
# ==========================================================================
COMMON_ENV = {
    "LLM_BASE_URL": "os.environ.get('LLM_BASE_URL', 'http://localhost:11434/v1')",
    "LLM_API_KEY":  "ollama",
    "LLM_MODEL":    "qwen3:32b",
    "NT_FILE":      NT_FILE,
    "POG_DATASET_FILE":  EXTRA_DATASET,
    "POG_SAMPLE_SIZE":   str(EXTRA_SIZE),
    "POG_SAMPLE_STRATEGY": "ordered",   # 子集已经按目标顺序排好
    "POG_MAX_DEPTH":     "3",
    "POG_FAST_MODE":     "0",
    "POG_PAPER_QA_CHAIN": "1",
    "POG_USE_LEGACY_KG_GATE": "0",
    "POG_RELATION_FILTER_THRESHOLD": "15",
    "POG_MIN_DEPTH_BEFORE_STOP":     "2",
    "POG_ACTIVE_CONFIG_NAMES": '["D_双模块全开"]',
    "POG_QUESTION_TIMEOUT": str(QUESTION_TIMEOUT),
}

RESULTS_DIR_GLOBAL = os.path.join(POG_DIR, "results")

def _snapshot():
    if not os.path.exists(RESULTS_DIR_GLOBAL): return set()
    return {fn for fn in os.listdir(RESULTS_DIR_GLOBAL) if fn.startswith("pog_4group_")}

def run_one_K(K):
    out_dir = os.path.join(OUTPUT_ROOT, f"K_{K}")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "console.log")
    before = _snapshot()
    env = os.environ.copy(); env.update(COMMON_ENV)
    env["POG_RELATION_TOPK"] = str(K)
    env["POG_CHECKPOINT_PATH"] = os.path.join(out_dir, "resume_checkpoint.json")

    print(f"\n{'='*70}\n[A1·E1] K = {K}  开始：{datetime.now():%H:%M:%S}\n{'='*70}\n")
    cmd = [sys.executable, os.path.join(POG_DIR, "run_experiment.py")]
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.Popen(cmd, cwd=POG_DIR, env=env, stdout=logf, stderr=subprocess.STDOUT)
        proc.wait()
    print(f"[A1·E1] K = {K}  完成（exit={proc.returncode}）  日志: {log_path}")
    after = _snapshot(); new_files = sorted(after - before)
    for fn in new_files:
        try:
            shutil.move(os.path.join(RESULTS_DIR_GLOBAL, fn), os.path.join(out_dir, fn))
            print(f"  [迁移] {fn} -> {out_dir}")
        except Exception as e:
            print(f"  [WARN] 迁移失败 {fn}: {e}")
    return out_dir

if __name__ == "__main__":
    if not os.path.exists(NT_FILE): print(f"[FATAL] KG 文件不存在: {NT_FILE}"); sys.exit(1)
    print(f"[A1 启动] 输出根目录: {OUTPUT_ROOT}")
    print(f"[A1 启动] 扩样集大小 {EXTRA_SIZE}，扫描 K = {K_VALUES}")
    t0 = time.time()
    for K in K_VALUES: run_one_K(K)
    print(f"\n[A1 完成] 总耗时 {(time.time()-t0)/3600:.2f} h；输出根目录: {OUTPUT_ROOT}")
