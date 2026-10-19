# NOTE: service address and credentials are read from environment
#       variables (LLM_BASE_URL / LLM_API_KEY / LLM_MODEL).
#       This file was sanitised before public release.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D′「判断但不停」实验补丁  ——  2026-10-16 DSH

作用：新增一个诊断配置组 D_prime_判断不停：保留充足性判断调用与日志，
      但忽略其"停止"结论，继续探索至最大深度。用于分离
      「判断调用改变了推理轨迹」与「真实提前停止」两种机制。

用法（在 notebook-509 容器内、pog_exp/ 目录下执行）：
    python3 d_prime_patch.py            # 应用（先 dry-run 校验锚点）
    python3 d_prime_patch.py --revert   # 从 .bak-dprime-* 还原

设计原则：
  · 幂等：已打过补丁会自动跳过
  · 安全：先全部校验锚点，全部命中才落盘；每步自动备份
  · 不改 PoG 主干逻辑：仅新增分支与统计，D 组原有行为不变
  · 原 judge 函数体不改，用包装器统一记录结论（覆盖全部 return 路径）
"""
import os
import re
import shutil
import sys
import datetime
import py_compile

BASE = os.path.dirname(os.path.abspath(__file__))
TS = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
REVERT = "--revert" in sys.argv
MARK = "[D-PRIME]"


def rd(p):
    return open(p, encoding="utf-8").read()


def wr(p, s):
    open(p, "w", encoding="utf-8").write(s)


def backup(p):
    b = f"{p}.bak-dprime-{TS}"
    shutil.copy2(p, b)
    return b


# ============================================================
# 补丁定义： (文件, 说明, 旧串, 新串)
# ============================================================
PATCHES = []

# ---------- 1) pog_improvements.py：新增 no_stop 配置 ----------
PATCHES.append((
    "pog_improvements.py",
    "CONFIG 增加 no_stop 开关",
    '''    "min_depth_before_stop":  2,
    # Prompt截断''',
    '''    "min_depth_before_stop":  2,
    # [D-PRIME] 判断但不停：保留判断调用，但忽略"停止"结论
    "no_stop":                False,
    # Prompt截断''',
))

# ---------- 2) pog_improvements.py：新增判断统计 ----------
PATCHES.append((
    "pog_improvements.py",
    "新增充足性判断的逐题统计与包装器",
    '''def set_config(key: str, value):
    CONFIG[key] = value
    print(f"[配置] {key} = {value}")''',
    '''def set_config(key: str, value):
    CONFIG[key] = value
    print(f"[配置] {key} = {value}")


# ========================================
# [D-PRIME] 充足性判断的逐题统计
# ========================================
JUDGE_STATS = {"judge_calls": 0, "judge_sufficient": 0,
               "judge_sufficient_at_depth": []}


def reset_judge_stats():
    """每题开始时调用"""
    JUDGE_STATS["judge_calls"] = 0
    JUDGE_STATS["judge_sufficient"] = 0
    JUDGE_STATS["judge_sufficient_at_depth"] = []


def get_judge_stats():
    """每题结束时读取；D' 与 D 的 judge_calls 应当相同"""
    s = dict(JUDGE_STATS)
    s["judge_sufficient_at_depth"] = list(JUDGE_STATS["judge_sufficient_at_depth"])
    s["no_stop"] = bool(CONFIG.get("no_stop", False))
    return s


def _note_judge(verdict, depth):
    JUDGE_STATS["judge_calls"] += 1
    if verdict:
        JUDGE_STATS["judge_sufficient"] += 1
        JUDGE_STATS["judge_sufficient_at_depth"].append(depth)''',
))

# ---------- 3) pog_improvements.py：原函数改名 + 包装 ----------
PATCHES.append((
    "pog_improvements.py",
    "judge_knowledge_sufficiency 改为包装器（记录全部 return 路径）",
    '''def judge_knowledge_sufficiency(
    paths_text: List[str] = None,''',
    '''def _judge_knowledge_sufficiency_impl(
    paths_text: List[str] = None,''',
))

PATCHES.append((
    "pog_improvements.py",
    "在 impl 之后插入同名包装器",
    '''# ========================================
# PoG论文：Path Summarizing（降低噪声 + 帮助评估/回答）
# ========================================''',
    '''def judge_knowledge_sufficiency(*args, **kwargs):
    """[D-PRIME] 包装 _judge_knowledge_sufficiency_impl，统一记录每次判断结论。

    用包装器而非改函数体，可覆盖原实现的全部 return 路径。
    对 D 组无行为影响（仅多一次计数）。
    """
    verdict = _judge_knowledge_sufficiency_impl(*args, **kwargs)
    depth = kwargs.get("current_depth")
    if depth is None and len(args) >= 4:
        depth = args[3]
    _note_judge(bool(verdict), depth)
    return verdict


# ========================================
# PoG论文：Path Summarizing（降低噪声 + 帮助评估/回答）
# ========================================''',
))

# ---------- 4) config.py：新增 D′ 配置组 ----------
PATCHES.append((
    "config.py",
    "EXPERIMENT_CONFIGS 增加 D_prime_判断不停",
    '''    {
        "name":      "D_双模块全开",
        "desc":      "PoG + 关系筛选与知识判断同时启用",
        "filter":    True,
        "judge":     True,
        "llm_filter": True,
        "llm_judge": True,
    },
]''',
    '''    {
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
]''',
))

# ---------- 5) run_experiment.py：透传 no_stop ----------
PATCHES.append((
    "run_experiment.py",
    "_apply_config 透传 no_stop",
    '''        set_config("min_depth_before_stop",  PAPER_MIN_DEPTH_BEFORE_STOP)''',
    '''        set_config("min_depth_before_stop",  PAPER_MIN_DEPTH_BEFORE_STOP)
        # [D-PRIME] 判断但不停开关（仅 D_prime 组为 True）
        set_config("no_stop", bool(cfg.get("no_stop", False)))''',
))

# ---------- 6) run_experiment.py：重置判断统计 ----------
PATCHES.append((
    "run_experiment.py",
    "每题开始时重置判断统计",
    '''        elapsed = round(time.time() - t0, 2)
        return {"success": success, "hop": hop or run_depth,''',
    '''        # [D-PRIME] 读取本题的判断统计（D 与 D' 的 judge_calls 应相同）
        try:
            from pog_improvements import get_judge_stats as _get_js
            _judge_stats = _get_js()
        except Exception as _e:
            _judge_stats = {"error": str(_e)}

        elapsed = round(time.time() - t0, 2)
        return {"success": success, "hop": hop or run_depth,''',
))

PATCHES.append((
    "run_experiment.py",
    "把判断统计写入逐题明细",
    '''                "top_paths_text": meta.get("top_paths_text", []),
                "error": None}''',
    '''                "top_paths_text": meta.get("top_paths_text", []),
                "judge_stats": _judge_stats,
                "error": None}''',
))

# ---------- 7) run_experiment.py：reset 调用 ----------
PATCHES.append((
    "run_experiment.py",
    "在 run_single 入口重置判断统计",
    '''def run_single(data, question, qid="Q"):
    topic_entity_dict, entity_ids = normalize_topic_entity(data.get("topic_entity", {}))''',
    '''def run_single(data, question, qid="Q"):
    # [D-PRIME] 逐题重置判断统计
    try:
        from pog_improvements import reset_judge_stats as _reset_js
        _reset_js()
    except Exception:
        pass
    topic_entity_dict, entity_ids = normalize_topic_entity(data.get("topic_entity", {}))''',
))

# ---------- 8) pog_core.py：no_stop 分支 ----------
PATCHES.append((
    "pog_core.py",
    "早停分支支持 no_stop（判断但不停止）",
    '''                if is_sufficient:
                    print(f"[PoG]   [改进-知识判断] 深度 {D}: 判断为充足，提前停止")
                    early_stop = True
                    final_hop = D
                    break''',
    '''                if is_sufficient:
                    if _pog_get_config("no_stop"):
                        # [D-PRIME] 判断照常发生并已记入统计，但不停止
                        print(f"[PoG]   [D' 判断不停] 深度 {D}: 判定充足，"
                              f"但 no_stop=True，继续探索（不早停）")
                    else:
                        print(f"[PoG]   [改进-知识判断] 深度 {D}: 判断为充足，提前停止")
                        early_stop = True
                        final_hop = D
                        break''',
))

# ---------- 9) pog_core.py：meta 带出判断统计 ----------
PATCHES.append((
    "pog_core.py",
    "meta 中带出判断统计",
    '''    meta = {
        "phase_stats": phase_stats,
        "top_paths_text": top_paths_text,
    }''',
    '''    meta = {
        "phase_stats": phase_stats,
        "top_paths_text": top_paths_text,
    }
    # [D-PRIME] 带出本题的判断统计（供 run_experiment 记录）
    try:
        from pog_improvements import get_judge_stats as _get_js2
        meta["judge_stats"] = _get_js2()
    except Exception:
        pass''',
))


# ============================================================
# 执行
# ============================================================
def do_revert():
    n = 0
    for f in ("pog_improvements.py", "config.py", "run_experiment.py", "pog_core.py"):
        p = os.path.join(BASE, f)
        cands = sorted([x for x in os.listdir(BASE) if x.startswith(f + ".bak-dprime-")])
        if not cands:
            print(f"  [跳过] {f}：无备份")
            continue
        shutil.copy2(os.path.join(BASE, cands[-1]), p)
        print(f"  [还原] {f}  ←  {cands[-1]}")
        n += 1
    print(f"\n  还原 {n} 个文件。")
    return 0


if REVERT:
    print("=== 还原 D′ 补丁 ===")
    sys.exit(do_revert())

print("=" * 74)
print("D′「判断但不停」补丁")
print("=" * 74)
print(f"  工作目录: {BASE}")
print(f"  时间戳  : {TS}")

# --- 校验所有锚点 ---
print("\n--- ① 锚点校验 ---")
files = {}
ok = True
for f, desc, old, new in PATCHES:
    p = os.path.join(BASE, f)
    if f not in files:
        if not os.path.exists(p):
            print(f"  ✗ 文件不存在: {f}")
            ok = False
            continue
        files[f] = rd(p)
    s = files[f]
    cnt = s.count(old)
    if cnt == 0 and new in s:
        print(f"  ○ [已打过] {f}: {desc}")
    elif cnt == 1:
        print(f"  ✓ {f}: {desc}")
    else:
        print(f"  ✗ {f}: {desc}  —— 锚点命中 {cnt} 次（应为 1）")
        ok = False

if not ok:
    print("\n  锚点校验失败，未做任何改动。")
    print("  请确认你运行的是 notebook-509 上的 pog_exp/ 副本（源码版本需与 2026-10-16 一致）。")
    sys.exit(1)

# --- 应用 ---
print("\n--- ② 应用补丁 ---")
for f in files:
    b = backup(os.path.join(BASE, f))
    print(f"  备份 {f} → {os.path.basename(b)}")

changed = set()
for f, desc, old, new in PATCHES:
    s = files[f]
    if old not in s:
        print(f"  ○ 跳过（已打过）{f}: {desc}")
        continue
    files[f] = s.replace(old, new, 1)
    changed.add(f)
    print(f"  ✓ {f}: {desc}")

for f in changed:
    wr(os.path.join(BASE, f), files[f])

# --- 语法检查 ---
print("\n--- ③ 语法检查 ---")
allok = True
for f in ("pog_improvements.py", "config.py", "run_experiment.py", "pog_core.py"):
    p = os.path.join(BASE, f)
    if not os.path.exists(p):
        continue
    try:
        py_compile.compile(p, doraise=True)
        print(f"  ✅ {f}")
    except Exception as e:
        print(f"  ❌ {f}: {e}")
        allok = False

if not allok:
    print("\n  ⚠️ 有语法错误，请执行：python3 d_prime_patch.py --revert")
    sys.exit(1)

# --- 冒烟验证 ---
print("\n--- ④ 冒烟验证 ---")
try:
    sys.path.insert(0, BASE)
    import importlib
    import pog_improvements as PI
    importlib.reload(PI)
    print(f"  no_stop 默认值        : {PI.CONFIG.get('no_stop')}")
    PI.reset_judge_stats()
    PI._note_judge(True, 2)
    PI._note_judge(False, 3)
    print(f"  统计自测              : {PI.get_judge_stats()}")
    print(f"  包装器存在            : {callable(PI.judge_knowledge_sufficiency)}")
    print(f"  原实现保留            : {callable(PI._judge_knowledge_sufficiency_impl)}")
    import config as CFG
    names = [c["name"] for c in CFG.EXPERIMENT_CONFIGS]
    print(f"  配置组                : {names}")
    print(f"  D_prime 在列          : {'D_prime_判断不停' in names}")
except Exception as e:
    print(f"  ⚠️ 冒烟验证异常（不阻塞）：{type(e).__name__}: {e}")

# --- 用法 ---
print("\n" + "=" * 74)
print("补丁完成。运行 D′ 实验：")
print("=" * 74)
print("""
  # 只跑 D′（与 D 同批更佳：也可一次跑 A/C/D/D′ 四组共享服务条件）
  export POG_DATASET_FILE=experiment_questions.json
  export NT_FILE=military_kg_group1.nt
  export POG_SAMPLE_SIZE=100
  export POG_RELATION_TOPK=10
  export POG_MIN_DEPTH_BEFORE_STOP=2
  export POG_ACTIVE_CONFIG_NAMES='["A_原始PoG","C_仅知识判断","D_双模块全开","D_prime_判断不停"]'
  python3 run_experiment.py

  结果位置： results/pog_4group_<时间戳>.{json,md}
  逐题明细中的 judge_stats 字段：
      judge_calls              判断调用次数（D 与 D′ 应相同）
      judge_sufficient         判定为"充足"的次数
      judge_sufficient_at_depth 判定充足时所在深度
  判读：
      D′ ≈ D  → 早停的"终止动作"无贡献，收益来自判断调用改变的推理轨迹
      D′ ≈ B  → 早停的终止有真实贡献
      D′ < B  → 判断调用本身有害
""")
print(f"  回滚：python3 d_prime_patch.py --revert   （备份后缀 .bak-dprime-{TS}）")
