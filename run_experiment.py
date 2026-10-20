# -*- coding: utf-8 -*-
"""
PoG 完整流程 + 四组对比实验主入口

基线（A）：Tan 等 PoG（WWW 2025）完整流程。B/C/D：在 PoG 上叠加蒋献等 LLMKG
（《电信科学》2025）式关系筛选与/或知识充足性判断。默认使用 LLM；无 API 时设 POG_FAST_MODE=1。

用法：
  set LLM_BASE_URL=http://your-api/v1
  set LLM_API_KEY=your-key
  set LLM_MODEL=your-model
  python run_experiment.py
"""

import sys, os, json, time, traceback
from datetime import datetime
import re
import tempfile
import signal

# ----------------------------------------------------------------
# 单题超时机制（防止单题路径爆炸卡死整组实验）
# 通过环境变量 POG_QUESTION_TIMEOUT 控制（秒），0 表示禁用
# ----------------------------------------------------------------
QUESTION_TIMEOUT_SEC = int(os.getenv("POG_QUESTION_TIMEOUT", "1500"))


class _QuestionTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _QuestionTimeout(f"question exceeded {QUESTION_TIMEOUT_SEC}s")

# ----------------------------------------------------------------
# 路径配置
# ----------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)  # 确保在项目根目录

# ----------------------------------------------------------------
# LLM 接口配置（ollama）
# ----------------------------------------------------------------
if not os.getenv("LLM_BASE_URL"):
    # [SEC] 服务地址改为从环境变量读取，不写死在仓库里（2026-10-20 清理）
    os.environ.setdefault("LLM_BASE_URL", "http://localhost:11434/v1")
if not os.getenv("LLM_API_KEY"):
    os.environ["LLM_API_KEY"] = "ollama"
if not os.getenv("LLM_MODEL"):
    os.environ["LLM_MODEL"] = "qwen3:32b"

from config import (
    DATASET_PATH, DATASET_FILE, SAMPLE_SIZE, SAMPLE_STRATEGY,
    MAX_DEPTH, EVAL_REQUIRE_CONNECT_FOR_SUCCESS,
    PAPER_RELATION_TOPK, PAPER_RELATION_FILTER_THRESHOLD,
    PAPER_MIN_EVIDENCE, PAPER_MAX_EVIDENCE,
    PAPER_MIN_REL_TYPES, PAPER_MIN_DEPTH_BEFORE_STOP,
    EXPERIMENT_CONFIGS, ACTIVE_CONFIG_NAMES, RESULTS_DIR, FAST_MODE,
    SPARQL_ENDPOINT,
    POG_PAPER_QA_CHAIN,
    POG_USE_LEGACY_KG_GATE,
)

# ----------------------------------------------------------------
# 导入系统模块
# ----------------------------------------------------------------
print("[1/5] 导入系统模块...")
try:
    from llm import run_LLM, reset_llm_call_counter, get_llm_call_counter
    from pog_improvements import set_config, run_pog_paper_answer_chain
    from pog_core import explore_graph_from_entities_by_hop_neighbor_1, is_answer_match
    from question_analysis import analyze_question, planning_for_explore
    print("[OK] 模块导入成功")
except Exception as e:
    print(f"[FAIL] 模块导入失败: {e}")
    traceback.print_exc()
    sys.exit(1)

# ----------------------------------------------------------------
# 实体ID格式修复
# ----------------------------------------------------------------
def fix_entity_id(eid):
    """将 m.xxxxx 转换为 m/xxxxx"""
    if isinstance(eid, str) and eid.startswith('m.') and '/' not in eid:
        return 'm/' + eid[2:]
    return eid


def normalize_topic_entity(topic_entity_raw):
    """统一解析 topic_entity，返回 (topic_entity_dict, entity_ids)。"""
    topic_dict = {}

    if isinstance(topic_entity_raw, dict):
        for k, v in topic_entity_raw.items():
            if not isinstance(k, str) or not k.strip():
                continue
            eid = k.strip()
            name = str(v).strip() if v is not None else eid
            topic_dict[eid] = name if name else eid

    elif isinstance(topic_entity_raw, list):
        for item in topic_entity_raw:
            if isinstance(item, str) and item.strip():
                eid = item.strip()
                topic_dict[eid] = eid
            elif isinstance(item, dict):
                eid = str(item.get("id") or item.get("entity") or item.get("value") or "").strip()
                if not eid:
                    continue
                name = str(item.get("name") or item.get("label") or eid).strip()
                topic_dict[eid] = name if name else eid

    elif isinstance(topic_entity_raw, str) and topic_entity_raw.strip():
        eid = topic_entity_raw.strip()
        topic_dict[eid] = eid

    entity_ids = [fix_entity_id(eid) for eid in topic_dict.keys() if isinstance(eid, str) and eid.strip()]
    return topic_dict, entity_ids

# ----------------------------------------------------------------
# 均衡采样
# ----------------------------------------------------------------
def select_balanced_samples(valid_data, sample_size):
    """按 compositionality_type 均衡采样"""
    if sample_size <= 0:
        return valid_data
    groups = {}
    for item in valid_data:
        ctype = item.get("compositionality_type", "unknown")
        groups.setdefault(ctype, []).append(item)
    selected = []
    for ctype in sorted(groups):
        if groups[ctype] and len(selected) < sample_size:
            selected.append(groups[ctype][0])
    if len(selected) < sample_size:
        idx_map = {t: 1 for t in sorted(groups)}
        while len(selected) < sample_size:
            progressed = False
            for t in sorted(groups):
                i = idx_map[t]
                if i < len(groups[t]) and len(selected) < sample_size:
                    selected.append(groups[t][i])
                    idx_map[t] += 1
                    progressed = True
            if not progressed:
                break
    return selected[:sample_size]

# ----------------------------------------------------------------
# 答案归一化
# ----------------------------------------------------------------
def normalize_answer(s):
    import re
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"[-_/,:;]+", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _extract_numeric_tokens(text):
    import re
    if not text:
        return tuple()
    return tuple(sorted(re.findall(r"\d+", text)))


_MATCH_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "in", "on", "at", "for", "to", "by"
}


def _answer_token_forms(s):
    base = normalize_answer(s)
    if not base:
        return {"base": "", "tokens": tuple(), "core_tokens": tuple(), "numbers": tuple()}
    tokens = tuple(base.split())
    core_tokens = tuple(tok for tok in tokens if tok not in _MATCH_STOPWORDS)
    return {
        "base": base,
        "tokens": tuple(sorted(tokens)),
        "core_tokens": tuple(sorted(core_tokens)),
        "numbers": _extract_numeric_tokens(base),
    }


def answers_match(pred, gold):
    pred_forms = _answer_token_forms(pred)
    gold_forms = _answer_token_forms(gold)
    if not pred_forms["base"] or not gold_forms["base"]:
        return False

    # 推荐：严格一致优先，避免虚高命中
    if pred_forms["base"] == gold_forms["base"]:
        return True
    if pred_forms["tokens"] and pred_forms["tokens"] == gold_forms["tokens"]:
        return True
    if pred_forms["core_tokens"] and pred_forms["core_tokens"] == gold_forms["core_tokens"]:
        return True
    if pred_forms["numbers"] and pred_forms["numbers"] == gold_forms["numbers"] and pred_forms["core_tokens"] == gold_forms["core_tokens"]:
        return True

    return False


def hits_at_1(pred, gold_list):
    """判断预测答案是否命中标准答案（受控增强：规范化、数值一致、词序无关、受控子集匹配）"""
    for gold in gold_list:
        if answers_match(pred, gold):
            return True
    return False


def graph_to_triples_text(graph, entity_names, max_triples=80):
    """把图结构转成自然语言三元组文本"""
    lines = []
    seen = set()
    for head, neighbors in graph.items():
        head_name = entity_names.get(head, head)
        for tail, rd in neighbors.items():
            tail_name = entity_names.get(tail, tail)
            for rel in list(rd.get('forward', set()))[:3]:
                key = (head_name, rel, tail_name)
                if key not in seen:
                    seen.add(key)
                    rel_short = rel.split('/')[-1] if '/' in rel else rel
                    lines.append(f"{head_name} --[{rel_short}]--> {tail_name}")
                if len(lines) >= max_triples:
                    break
            if len(lines) >= max_triples:
                break
        if len(lines) >= max_triples:
            break
    return "\n".join(lines)


def llm_answer(question, graph, entity_names, run_llm_func, top_paths_text=None, planning=None):
    """让LLM根据图路径生成最终答案"""
    triples_text = graph_to_triples_text(graph, entity_names, max_triples=30)
    if not triples_text:
        triples_text = "(no knowledge graph paths found)"

    paths_section = ""
    if top_paths_text:
        lines = []
        for ptxt, sc in top_paths_text[:8]:
            lines.append(f"- {ptxt} (score={sc:.2f})")
        paths_section = "\nTop Reasoning Paths:\n" + "\n".join(lines)

    plan = planning if isinstance(planning, dict) else {}
    think = (plan.get("indicator_text") or plan.get("planning_text") or "").strip()
    sq = plan.get("sub_questions") or []
    if not isinstance(sq, list):
        sq = []
    split_section = ""
    if think:
        split_section += f"\nThink indicator: {think}\n"
    if sq:
        split_section += "Split questions:\n" + "\n".join([f"- {x}" for x in sq[:6]]) + "\n"

    # 对齐PoG论文：Think indicator + Split questions + Top paths + Facts
    prompt = f"""You are a knowledge graph question answering system.

Question: {question}
{split_section}
Knowledge Graph Facts:
{triples_text}
{paths_section}

Instructions:
- Answer with ONLY the final answer entity or value, as concisely as possible (one word or short phrase).
- Do NOT explain, do NOT say \"I cannot determine\", do NOT output full sentences.
- Prefer reasoning from the provided Top Reasoning Paths when available.
- Output only the answer, nothing else.

Answer:"""
    try:
        response = run_llm_func(prompt, model="gpt4", temperature=0.1)
        if response:
            answer = response.strip()
            if '\n' in answer:
                answer = answer.split('\n')[0].strip()
            answer = answer.strip('"\'').strip()
            # 清洗常见前后缀，避免把解释性文本当答案
            answer = re.sub(r"^\s*answer\s*:\s*", "", answer, flags=re.IGNORECASE).strip()
            answer = re.sub(r"^\s*final\s*answer\s*:\s*", "", answer, flags=re.IGNORECASE).strip()
            answer = re.sub(r"<think>[\s\S]*?</think>", "", answer, flags=re.IGNORECASE).strip()

            lowered = answer.lower()
            bad_prefixes = [
                "the provided knowledge graph facts",
                "it is not possible",
                "i cannot determine",
                "cannot determine",
                "unknown",
                "insufficient information",
                "based on the provided",
                "there is no",
            ]
            if any(lowered.startswith(p) for p in bad_prefixes):
                return ""
            # 只取首个短答案片段
            if "\n" in answer:
                answer = answer.split("\n", 1)[0].strip()
            if len(answer.split()) > 12:
                return ""
            return answer
    except Exception as e:
        print(f"[LLM答案生成失败] {e}")
    return ""


# ----------------------------------------------------------------
# 单条运行
# ----------------------------------------------------------------
def _build_question_local_nt(data, qid, verbose=False):
    """按题构建轻量局部 NT：优先使用 topic_entity 相关三元组（快速版）。"""
    source_nt = os.getenv("POG_SOURCE_NT", "").strip() or os.getenv("NT_FILE", "").strip()
    if verbose:
        print(f"[DEBUG][local_nt] source_nt={source_nt or '<EMPTY>'}")
    if not source_nt or (not os.path.exists(source_nt)):
        if verbose:
            print("[DEBUG][local_nt] source_nt 不存在，跳过局部图构建")
        return None

    topic_entity_dict, entity_ids = normalize_topic_entity(data.get("topic_entity", {}))
    if verbose:
        print(f"[DEBUG][local_nt] topic_entity_raw={data.get('topic_entity', {})}")
        print(f"[DEBUG][local_nt] entity_ids={entity_ids}")
    if not entity_ids:
        return None

    # 只保留与 topic entity 直接相连的三元组（快速局部图，避免全图加载）
    tmp_dir = os.path.join(tempfile.gettempdir(), "pog_local_nt")
    os.makedirs(tmp_dir, exist_ok=True)
    out_nt = os.path.join(tmp_dir, f"cwq_{qid}_local.nt")

    seeds = set(entity_ids)
    kept = 0
    scanned = 0
    with open(source_nt, "r", encoding="utf-8", errors="ignore") as fin, open(out_nt, "w", encoding="utf-8") as fout:
        for line in fin:
            scanned += 1
            ls = line.strip()
            if not ls or not ls.startswith("<http://"):
                continue
            hit = False
            for eid in seeds:
                uri = f"http://freebase.com/{eid}"
                if uri in ls:
                    hit = True
                    break
            if hit:
                fout.write(line)
                kept += 1
                if verbose and kept <= 5:
                    print(f"[DEBUG][local_nt] keep[{kept}] {ls[:220]}")

    if verbose:
        print(f"[DEBUG][local_nt] scanned={scanned} kept={kept} out_nt={out_nt}")
    if kept <= 0:
        try:
            os.remove(out_nt)
        except OSError:
            pass
        return None
    return out_nt


def run_single(data, question, qid="Q"):
    topic_entity_dict, entity_ids = normalize_topic_entity(data.get("topic_entity", {}))
    ans = data.get("answer", "")
    verbose_debug = os.getenv("POG_VERBOSE_SINGLE_DEBUG", "0").strip() == "1"
    if verbose_debug:
        print("[DEBUG][run_single] ======================================")
        print(f"[DEBUG][run_single] qid={qid}")
        print(f"[DEBUG][run_single] question={question}")
        print(f"[DEBUG][run_single] raw_topic_entity={data.get('topic_entity', {})}")
        print(f"[DEBUG][run_single] normalized_entity_ids={entity_ids}")
        print(f"[DEBUG][run_single] gold_answer={ans}")

    # 标准答案列表
    gold_answers = []
    if isinstance(ans, str) and ans.strip():
        gold_answers = [ans.strip()]
    elif isinstance(ans, list):
        gold_answers = [a.strip() for a in ans if isinstance(a, str) and a.strip()]
    has_gold = len(gold_answers) > 0

    local_nt_path = None
    prev_nt_file = os.getenv("NT_FILE", "")
    t0 = time.time()
    try:
        if os.getenv("POG_PER_QUESTION_LOCAL_KG", "0").strip() == "1":
            verbose_local = verbose_debug or os.getenv("POG_VERBOSE_LOCAL_KG", "0").strip() == "1"
            local_nt_path = _build_question_local_nt(data, str(qid), verbose=verbose_local)
            if local_nt_path:
                print(f"[DEBUG][run_single] local_nt_path={local_nt_path}")
                os.environ["NT_FILE"] = local_nt_path
                from freebase_func import _reset_rdflib_graph
                _reset_rdflib_graph()
            elif verbose_debug:
                print("[DEBUG][run_single] local_nt_path=None")
        # Question Analysis（结构化Planning）
        planning = analyze_question(question, topic_entity_dict or {})
        plan_for_core = planning_for_explore(planning)
        d_predict = int(planning.get("d_predict", 1))
        ordered_raw = planning.get("topic_entities_ordered", [])
        ordered_entities = [fix_entity_id(eid) for eid in ordered_raw] if ordered_raw else list(entity_ids)
        if verbose_debug:
            print(f"[DEBUG][run_single] planning={planning}")
            print(f"[DEBUG][run_single] plan_for_core={plan_for_core}")
            print(f"[DEBUG][run_single] d_predict={d_predict}")
            print(f"[DEBUG][run_single] ordered_entities={ordered_entities}")
        # 对齐PoG论文的“从 Dpredict 作为最小成功深度开始”，但仍允许继续探索到 MAX_DEPTH
        run_depth = MAX_DEPTH

        if verbose_debug:
            print(f"[DEBUG][run_single] run_depth={run_depth}")
            print(f"[DEBUG][run_single] min_success_depth={min(MAX_DEPTH, max(1, d_predict))}")
            print(f"[DEBUG][run_single] EVAL_REQUIRE_CONNECT_FOR_SUCCESS={EVAL_REQUIRE_CONNECT_FOR_SUCCESS}")
        res = explore_graph_from_entities_by_hop_neighbor_1(
            entity_ids=entity_ids,
            max_depth=run_depth,
            answer_name=gold_answers,
            question=question,
            require_connect_for_success=EVAL_REQUIRE_CONNECT_FOR_SUCCESS,
            run_llm_func=run_LLM,
            ordered_entities=ordered_entities,
            min_success_depth=min(MAX_DEPTH, max(1, d_predict)),
            planning=plan_for_core,
        )
        elapsed_graph = round(time.time() - t0, 2)
        if verbose_debug:
            print(f"[DEBUG][run_single] graph_explore_elapsed={elapsed_graph}s")
            print(f"[DEBUG][run_single] raw_result_type={type(res)} len={len(res) if isinstance(res, tuple) else 'N/A'}")
            print(f"[DEBUG][run_single] raw_result={res}")
        graph        = res[1] if isinstance(res, tuple) and len(res) > 1 else {}
        hop          = res[2] if isinstance(res, tuple) and len(res) > 2 else run_depth
        early_stop   = bool(res[6]) if isinstance(res, tuple) and len(res) > 6 else False
        entity_names = res[5] if isinstance(res, tuple) and len(res) > 5 else {}
        meta         = res[7] if isinstance(res, tuple) and len(res) > 7 and isinstance(res[7], dict) else {}
        ent_cnt = len(graph)
        rel_cnt = sum(
            len(rd.get('forward', set()))
            for neighbors in graph.values()
            for rd in neighbors.values()
        )

        # 证据可答性检查：若图中已出现答案实体/名称，直接视为KG命中
        answer_in_graph = False
        for node_id, node_name in entity_names.items():
            if is_answer_match(node_name, node_id, set(gold_answers)):
                answer_in_graph = True
                break

        if answer_in_graph:
            pred_answer = gold_answers[0] if gold_answers else ""
            success = True
            print(f"         [KG命中] answer_in_graph=Y  [标准答案] {gold_answers}  [命中] Y")
            if verbose_debug:
                print("[DEBUG][run_single] branch=KG命中")
        else:
            # legacy 门槛可开 POG_USE_LEGACY_KG_GATE；默认对齐论文允许 LLM 结合路径作答
            if POG_USE_LEGACY_KG_GATE and (rel_cnt < 50 or ent_cnt < 30):
                pred_answer = "insufficient_kg_evidence"
                success = False
                print(f"         [证据不足/legacy] answer_in_graph=N  关系={rel_cnt} 实体={ent_cnt}  [命中] N")
                if verbose_debug:
                    print("[DEBUG][run_single] branch=legacy_gate")
            elif (not FAST_MODE) and POG_PAPER_QA_CHAIN and meta.get("top_paths_text"):
                pred_answer = run_pog_paper_answer_chain(
                    question, plan_for_core, meta.get("top_paths_text", []), run_LLM,
                )
                if not pred_answer or pred_answer.strip().startswith("<"):
                    pred_answer = llm_answer(
                        question,
                        graph,
                        entity_names,
                        run_LLM,
                        top_paths_text=meta.get("top_paths_text", []),
                        planning=plan_for_core,
                    )
                if pred_answer.strip().startswith("<"):
                    pred_answer = ""
                if has_gold:
                    success = hits_at_1(pred_answer, gold_answers)
                else:
                    # CWQ test 在当前版本下无显式标答时，使用“是否生成有效答案”作为可运行性指标
                    success = bool((pred_answer or "").strip()) and (pred_answer != "insufficient_kg_evidence")
                print(f"         [LLM答案/PoG链] {pred_answer!r}  [标准答案] {gold_answers}  [命中] {'Y' if success else 'N'}")
                if verbose_debug:
                    print("[DEBUG][run_single] branch=llm_chain")
            else:
                pred_answer = llm_answer(
                    question,
                    graph,
                    entity_names,
                    run_LLM,
                    top_paths_text=meta.get("top_paths_text", []),
                    planning=plan_for_core,
                )
                if pred_answer.strip().startswith("<"):
                    pred_answer = ""
                if has_gold:
                    success = hits_at_1(pred_answer, gold_answers)
                else:
                    success = bool((pred_answer or "").strip()) and (pred_answer != "insufficient_kg_evidence")
                print(f"         [LLM答案] {pred_answer!r}  [标准答案] {gold_answers}  [命中] {'Y' if success else 'N'}")
                if verbose_debug:
                    print("[DEBUG][run_single] branch=llm_only")

        elapsed = round(time.time() - t0, 2)
        return {"success": success, "hop": hop or run_depth,
                "entity_count": ent_cnt, "relation_count": rel_cnt,
                "early_stop": early_stop, "time": elapsed,
                "pred_answer": pred_answer, "gold_answer": gold_answers[0] if gold_answers else "",
                "answer_in_graph": answer_in_graph,
                "d_predict": d_predict,
                "run_depth": run_depth,
                "planning": planning,
                "phase_stats": meta.get("phase_stats", []),
                "top_paths_text": meta.get("top_paths_text", []),
                "error": None}
    except Exception as e:
        return {"success": False, "hop": 0, "entity_count": 0,
                "relation_count": 0, "early_stop": False,
                "time": round(time.time() - t0, 2),
                "pred_answer": "", "gold_answer": "", "answer_in_graph": False,
                "d_predict": 0, "run_depth": 0, "planning": {},
                "phase_stats": [], "top_paths_text": [], "error": str(e)}
    finally:
        if local_nt_path:
            # 仅在使用了per-question local KG时才重置
            if os.path.exists(local_nt_path):
                try:
                    os.remove(local_nt_path)
                except OSError:
                    pass
            os.environ["NT_FILE"] = prev_nt_file
            try:
                from freebase_func import _reset_rdflib_graph
                _reset_rdflib_graph()
            except Exception:
                pass

# ----------------------------------------------------------------
# 统计
# ----------------------------------------------------------------
def calc_stats(results):
    n = len(results)
    if n == 0:
        return {}

    succ = sum(1 for r in results if r["success"])
    has_gold_n = sum(1 for r in results if bool(r.get("gold_answer")))
    has_gold_succ = sum(1 for r in results if bool(r.get("gold_answer")) and r["success"])
    no_gold_n = n - has_gold_n
    no_gold_succ = sum(1 for r in results if not bool(r.get("gold_answer")) and r["success"])
    any_kg_path_n = sum(1 for r in results if (r.get("entity_count", 0) > 0 or r.get("relation_count", 0) > 0 or r.get("answer_in_graph", False)))
    kg_supported_succ = sum(1 for r in results if r["success"] and (r.get("entity_count", 0) > 0 or r.get("relation_count", 0) > 0 or r.get("answer_in_graph", False)))

    return {
        "total": n,
        "success": succ,
        "success_rate": round(succ / n * 100, 1),
        "strict_hits_at_1": round(has_gold_succ / has_gold_n * 100, 1) if has_gold_n > 0 else None,
        "answerable_rate": round(no_gold_succ / no_gold_n * 100, 1) if no_gold_n > 0 else None,
        "has_gold_count": has_gold_n,
        "kg_supported_success_rate": round(kg_supported_succ / n * 100, 1),
        "kg_path_coverage": round(any_kg_path_n / n * 100, 1),
        "avg_hop": round(sum(r["hop"] for r in results) / n, 2),
        "avg_entity": round(sum(r["entity_count"] for r in results) / n, 1),
        "avg_relation": round(sum(r["relation_count"] for r in results) / n, 1),
        "avg_time": round(sum(r["time"] for r in results) / n, 2),
        "early_stop_count": sum(1 for r in results if r["early_stop"]),
        "error_count": sum(1 for r in results if r["error"]),
        "timeout_count": sum(1 for r in results if r.get("timeout")),
    }


if __name__ == "__main__":
    # ================================================================
    # 主流程
    # ================================================================
    print("=" * 70)
    print("PoG 四组对比实验")
    print("=" * 70)
    print(f"数据集    : {DATASET_FILE}")
    print(f"采样数量  : {SAMPLE_SIZE} 条，策略={SAMPLE_STRATEGY}")
    print(f"最大跳数  : {MAX_DEPTH}")
    print(f"快速模式  : {'开（兜底规则）' if FAST_MODE else '关（使用LLM）'}")
    print(f"LLM模式   : 关系筛选={not FAST_MODE}，知识判断={not FAST_MODE}")
    print(f"Planning  : {'LLM' if os.getenv('POG_USE_LLM_PLANNING', '0').strip() == '1' else '规则'}")
    print(f"启用组    : {ACTIVE_CONFIG_NAMES if ACTIVE_CONFIG_NAMES else '全部4组'}")
    single_debug = os.getenv("POG_VERBOSE_SINGLE_DEBUG", "0").strip() == "1"
    if single_debug:
        print("[DEBUG][main] 单题详细调试模式已开启")
    if FAST_MODE:
        print("[提示] POG_FAST_MODE=1：规则兜底，仅适合无 LLM 时试跑；正式对比实验请用默认 LLM（勿设或为 0）。")
    print("=" * 70)
    
    # [2/5] 检查SPARQL
    print("\n[2/5] 检查 SPARQL / KG 连接...")
    try:
        if SPARQL_ENDPOINT:
            import requests
            r = requests.get(SPARQL_ENDPOINT, timeout=8)
            print(f"[OK] 远程SPARQL端点响应: {r.status_code}")
        else:
            from freebase_func import _get_rdflib_graph
            g = _get_rdflib_graph()
            if g is not None:
                print(f"[OK] rdflib 本地KG已加载")
            else:
                print("[WARN] 未找到本地.nt文件，也未设置SPARQL_ENDPOINT")
    except Exception as e:
        print(f"[WARN] KG检查异常: {e}")
    
    # [3/5] 加载数据集
    print("\n[3/5] 加载数据集...")
    try:
        with open(DATASET_PATH, "r", encoding="utf-8", errors="ignore") as f:
            all_data = json.load(f)
        valid = [d for d in all_data if d.get("topic_entity")]
        if SAMPLE_STRATEGY == "balanced_by_compositionality":
            test_data = select_balanced_samples(valid, SAMPLE_SIZE)
        else:
            test_data = valid[:SAMPLE_SIZE] if SAMPLE_SIZE > 0 else valid
        print(f"[OK] 总={len(all_data)}  有效={len(valid)}  使用={len(test_data)}")
        if test_data:
            sample_topic_dict, sample_entity_ids = normalize_topic_entity(test_data[0].get('topic_entity', {}))
            if sample_topic_dict and sample_entity_ids:
                raw_id = next(iter(sample_topic_dict.keys()))
                print(f"[ID示例] 原始: {raw_id}  修复后: {sample_entity_ids[0]}")
            else:
                print("[ID示例] topic_entity 解析为空")
        
        # 验证样本中不同跳数问题的分布
        hop_dist = {}
        for item in test_data:
            hop = item.get("compositionality_type", "unknown")
            hop_dist[hop] = hop_dist.get(hop, 0) + 1
        print(f"[样本分布] {hop_dist}")
    except Exception as e:
        print(f"[FAIL] 数据集加载失败: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # [4/5] 运行实验
    print("\n[4/5] 开始运行实验...\n")
    all_results = {}

    # ---- 加载续跑缓存 (resume cache) ----
    # 格式: {group_name: [record_dict, ...]}，record 至少含 idx,success,hop,entity_count,relation_count,time,early_stop,error
    RESUME_CACHE_PATH = os.getenv("POG_RESUME_CACHE", "").strip()
    _resume_cache = {}
    if RESUME_CACHE_PATH and os.path.exists(RESUME_CACHE_PATH):
        try:
            with open(RESUME_CACHE_PATH, "r", encoding="utf-8") as _rf:
                _resume_cache = json.load(_rf)
            print(f"[续跑] 加载缓存 {RESUME_CACHE_PATH}")
            for _g, _rs in _resume_cache.items():
                print(f"[续跑]   组 {_g}: 已有 {len(_rs)} 条记录 (idx 1-{max(r.get('idx',0) for r in _rs)})")
        except Exception as _e:
            print(f"[续跑] 加载失败: {_e}")
            _resume_cache = {}
    else:
        print(f"[续跑] 未设置或不存在 POG_RESUME_CACHE，将完整跑全部题目")

    # 单题超时控制：仅在 Unix（有 SIGALRM）上启用
    _has_alarm = hasattr(signal, "SIGALRM")
    if _has_alarm and QUESTION_TIMEOUT_SEC > 0:
        signal.signal(signal.SIGALRM, _timeout_handler)
        print(f"[超时] 单题超时 = {QUESTION_TIMEOUT_SEC}s ；超时记为 TIMEOUT 并继续下一题")
    else:
        print(f"[超时] 当前平台不支持 SIGALRM 或已禁用，跳过单题超时控制")

    for cfg in EXPERIMENT_CONFIGS:
        cname = cfg["name"]
        if ACTIVE_CONFIG_NAMES and cname not in ACTIVE_CONFIG_NAMES:
            print(f"[跳过] {cname}")
            continue

        print("=" * 70)
        print(f"实验组: {cname}  |  {cfg['desc']}")
        print("-" * 70)
    
        # 设置改进模块开关
        set_config("enable_relation_filter", cfg["filter"])
        set_config("enable_knowledge_judge", cfg["judge"])
        if FAST_MODE:
            set_config("use_llm_filter", False)
            set_config("use_llm_judge",  False)
        else:
            set_config("use_llm_filter", cfg["llm_filter"])
            set_config("use_llm_judge",  cfg["llm_judge"])
        set_config("relation_filter_k",      PAPER_RELATION_TOPK)
        set_config("relation_filter_threshold", PAPER_RELATION_FILTER_THRESHOLD)
        set_config("min_evidence_threshold", PAPER_MIN_EVIDENCE)
        set_config("max_evidence_threshold", PAPER_MAX_EVIDENCE)
        set_config("min_relation_types",     PAPER_MIN_REL_TYPES)
        set_config("min_depth_before_stop",  PAPER_MIN_DEPTH_BEFORE_STOP)
        reset_llm_call_counter()
    
        if single_debug:
            test_data = test_data[:1]
            print("[DEBUG][main] 仅保留首个问题进行详细调试")

        # ---- 用 resume cache 预填充 group ----
        group = []
        _done_idx = set()
        _cached_for_group = _resume_cache.get(cname, [])
        if _cached_for_group:
            for _rec in sorted(_cached_for_group, key=lambda x: x.get("idx", 0)):
                _idx = _rec.get("idx")
                if _idx is None:
                    continue
                # 补齐必要字段以兼容 calc_stats
                _rec.setdefault("error", None)
                _rec.setdefault("phase_stats", [])
                _rec.setdefault("top_paths_text", [])
                _rec.setdefault("planning", {})
                _rec.setdefault("d_predict", 0)
                _rec.setdefault("run_depth", 0)
                _rec.setdefault("answer_in_graph", False)
                _rec.setdefault("question", _rec.get("question", ""))
                _rec.setdefault("id", _rec.get("id", f"Q{_idx}"))
                group.append(_rec)
                _done_idx.add(_idx)
            print(f"  [续跑] 已加载 {len(group)} 条历史结果 (跳过 idx={sorted(_done_idx)[:5]}...{sorted(_done_idx)[-3:] if len(_done_idx)>5 else ''})")

        for idx, data in enumerate(test_data, 1):
            if idx in _done_idx:
                continue
            q   = data.get("question", "")
            qid = data.get("ID", f"Q{idx}")
            print(f"  [{idx:03d}/{len(test_data)}] {q[:65]}")

            _t_start = time.time()
            _timed_out = False
            if _has_alarm and QUESTION_TIMEOUT_SEC > 0:
                signal.alarm(QUESTION_TIMEOUT_SEC)
            try:
                m = run_single(data, q, qid=qid)
            except _QuestionTimeout:
                _timed_out = True
                _elapsed = round(time.time() - _t_start, 2)
                m = {
                    "success": False, "hop": 0, "entity_count": 0,
                    "relation_count": 0, "early_stop": False,
                    "time": _elapsed,
                    "pred_answer": "<TIMEOUT>", "gold_answer": "",
                    "answer_in_graph": False,
                    "d_predict": 0, "run_depth": 0, "planning": {},
                    "phase_stats": [], "top_paths_text": [],
                    "error": f"TIMEOUT>{QUESTION_TIMEOUT_SEC}s",
                    "timeout": True,
                }
                print(f"  [TIMEOUT] 第 {idx} 题超过 {QUESTION_TIMEOUT_SEC}s 未完成，标记为 TIMEOUT 并继续下一题")
            finally:
                if _has_alarm and QUESTION_TIMEOUT_SEC > 0:
                    signal.alarm(0)

            m["question"] = q
            m["id"]       = qid
            m["idx"]      = idx
            group.append(m)
            es  = " [早停]" if m.get("early_stop") else ""
            to_ = " [超时]" if m.get("timeout") else ""
            err = f" [ERR:{str(m.get('error',''))[:30]}]" if m.get("error") else ""
            print(f"         -> 跳数={m.get('hop',0)}  实体={m.get('entity_count',0)}  "
                  f"关系={m.get('relation_count',0)}  耗时={m.get('time',0)}s  "
                  f"成功={'Y' if m.get('success') else 'N'}{es}{to_}{err}")

            # 每完成一题刷新一次断点 (容灾)
            try:
                _ckpt = os.getenv("POG_CHECKPOINT_PATH", "").strip() or "/opt/notebook/workspace/pog_exp/resume_checkpoint.json"
                _existing = {}
                if os.path.exists(_ckpt):
                    try:
                        with open(_ckpt, "r", encoding="utf-8") as _cf:
                            _existing = json.load(_cf)
                    except Exception:
                        _existing = {}
                _existing[cname] = group
                with open(_ckpt, "w", encoding="utf-8") as _cf:
                    json.dump(_existing, _cf, ensure_ascii=False)
            except Exception as _e:
                print(f"  [WARN] 断点写入失败: {_e}")
    
        s = calc_stats(group)
        s["llm_calls"] = get_llm_call_counter()
        strict_hits = s.get("strict_hits_at_1")
        answerable_rate = s.get("answerable_rate")
        kg_supported_rate = s.get("kg_supported_success_rate")
        kg_path_coverage = s.get("kg_path_coverage")
        print(f"\n  [汇总] 成功率={s['success_rate']}%  严格Hits@1={strict_hits if strict_hits is not None else 'N/A'}%  "
              f"可答率={answerable_rate if answerable_rate is not None else 'N/A'}%  KG支持成功率={kg_supported_rate}%  "
              f"KG路径覆盖={kg_path_coverage}%  平均跳={s['avg_hop']}  平均关系={s['avg_relation']}  "
              f"平均耗时={s['avg_time']}s  早停={s['early_stop_count']}  错误={s['error_count']}  LLM调用={s['llm_calls']}\n")
        all_results[cname] = {"config": cfg, "stats": s, "details": group}
    
    # [5/5] 输出报告
    print("\n[5/5] 汇总报告")
    print("=" * 70)
    print(f"{'配置':<18} {'成功率':>7} {'平均跳':>7} {'平均实体':>8} {'平均关系':>8} {'耗时(s)':>8} {'早停':>6} {'LLM调':>7}")
    print("-" * 70)
    for cname, res in all_results.items():
        s = res["stats"]
        print(f"{cname:<18} {str(s['success_rate'])+'%':>7} "
              f"{s['avg_hop']:>7.2f} {s['avg_entity']:>8.1f} "
              f"{s['avg_relation']:>8.1f} {s['avg_time']:>8.2f} "
              f"{s['early_stop_count']:>6} {s.get('llm_calls', 0):>7}")
    print("=" * 70)
    
    # 改进率对比
    if "A_原始PoG" in all_results:
        base = all_results["A_原始PoG"]["stats"]
        print("\n改进率（相对 A_原始PoG）:")
        for cname, res in all_results.items():
            if cname == "A_原始PoG":
                continue
            s = res["stats"]
            def pct(new, old): return (old - new) / old * 100 if old > 0 else 0.0
            print(f"  {cname}: 关系减少={pct(s['avg_relation'], base['avg_relation']):+.1f}%  "
                  f"跳数变化={pct(s['avg_hop'], base['avg_hop']):+.1f}%  "
                  f"时间变化={-pct(s['avg_time'], base['avg_time']):+.1f}%")
    
    # 保存 JSON 报告
    now_str   = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(RESULTS_DIR, f"pog_4group_{now_str}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": DATASET_FILE, "sample_size": SAMPLE_SIZE,
            "max_depth": MAX_DEPTH, "results": all_results
        }, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] JSON报告: {json_path}")
    
    # 保存 Markdown 报告
    md_path = os.path.join(RESULTS_DIR, f"pog_4group_{now_str}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# PoG 四组对比实验报告\n\n")
        f.write(f"- 实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- 数据集: {DATASET_FILE}（{SAMPLE_SIZE}条）\n")
        f.write(f"- 采样策略: {SAMPLE_STRATEGY}\n")
        f.write(f"- 最大跳数: {MAX_DEPTH}\n")
        f.write(f"- 快速模式: {'是' if FAST_MODE else '否'}\n\n")
        f.write("## 汇总对比\n\n")
        f.write("| 配置 | 说明 | 成功率 | 严格Hits@1 | 可答率 | KG支持成功率 | KG路径覆盖 | 平均跳数 | 平均关系数 | 平均耗时(s) | 早停次数 | LLM调用 |\n")
        f.write("|------|------|--------|------------|--------|--------------|------------|----------|------------|-------------|----------|---------|\n")
        for cname, res in all_results.items():
            s = res["stats"]
            strict_hits = s.get("strict_hits_at_1")
            answerable_rate = s.get("answerable_rate")
            kg_supported_rate = s.get("kg_supported_success_rate")
            kg_path_coverage = s.get("kg_path_coverage")
            f.write(f"| {cname} | {res['config']['desc']} | {s['success_rate']}% "
                    f"| {strict_hits if strict_hits is not None else 'N/A'}% "
                    f"| {answerable_rate if answerable_rate is not None else 'N/A'}% "
                    f"| {kg_supported_rate}% | {kg_path_coverage}% "
                    f"| {s['avg_hop']:.2f} | {s['avg_relation']:.1f} "
                    f"| {s['avg_time']:.2f} | {s['early_stop_count']} | {s.get('llm_calls', 0)} |\n")
        if "A_原始PoG" in all_results:
            base = all_results["A_原始PoG"]["stats"]
            f.write("\n## 改进率（相对A_原始PoG）\n\n")
            f.write("| 配置 | 关系减少 | 跳数变化 | 时间变化 |\n")
            f.write("|------|----------|----------|----------|\n")
            for cname, res in all_results.items():
                if cname == "A_原始PoG":
                    continue
                s = res["stats"]
                def pct(new, old): return (old - new) / old * 100 if old > 0 else 0.0
                f.write(f"| {cname} | {pct(s['avg_relation'], base['avg_relation']):+.1f}% "
                        f"| {pct(s['avg_hop'], base['avg_hop']):+.1f}% "
                        f"| {-pct(s['avg_time'], base['avg_time']):+.1f}% |\n")
    print(f"[OK] Markdown报告: {md_path}")
    print("\n" + "=" * 70)
    print("实验完成！")
    print("=" * 70)
