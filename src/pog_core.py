# NOTE: service address and credentials are read from environment
#       variables (LLM_BASE_URL / LLM_API_KEY / LLM_MODEL).
#       This file was sanitised before public release.
# -*- coding: utf-8 -*-
"""
PoG (Paths-over-Graph) 完整实现 — 对齐 WWW'25 论文算法

四大阶段：
  Phase 1  Initialization    — Question Analysis + Subgraph Detection
  Phase 2  Exploration       — 三阶段路径探索
           2a  Topic Entity Path Exploration
           2b  LLM Supplement Path Exploration
           2c  Node Expand Exploration
  Phase 3  Path Pruning      — 三步 Beam Search（Branch Reduced → Fuzzy → Precise）
  Phase 4  Question Answering — Path Summarizing → QA Evaluation → Final Answering

改进模块插入点（B/C/D 组）：
  - 关系筛选 Top-K  → 每跳关系展开前（filter_relations_topk）
  - 知识充足性判断  → 每跳展开后（judge_knowledge_sufficiency）
"""

import re
import json
import time
import math
from typing import Dict, List, Set, Tuple, Optional, Any

from freebase_func import get_1hop_neighbors, get_name

# ---- 改进模块（可选） ----
try:
    from pog_improvements import (
        filter_relations_topk,
        judge_knowledge_sufficiency,
        summarize_paths_for_qa,
        get_config as _pog_get_config,
    )
    _IMPROVEMENTS_OK = True
except Exception:
    _IMPROVEMENTS_OK = False
    def _pog_get_config(key):
        return False

# ---- 配置 ----
try:
    from config import POG_W1_FUZZY, POG_WMAX, FAST_MODE
except Exception:
    POG_W1_FUZZY, POG_WMAX, FAST_MODE = 80, 3, False

# ---- SBERT（Fuzzy Selection 用） ----
_sbert_model = None

def _collect_judge_stats():
    """[D-PRIME][FIX-9] 收集本题的判断统计。

    保证 judge_stats 字段【始终存在】：正常时返回真实统计；
    失败时打印告警并返回结构完整的空统计（带 _error 标记），
    避免结果 JSON 静默缺字段导致后续 D/D' 分析得到错误的"无数据"结论。
    返回键与 pog_improvements.get_judge_stats() 保持一致。
    """
    try:
        from pog_improvements import get_judge_stats as _get_js
        return _get_js()
    except Exception as _e:
        import traceback as _tb
        print(f"[WARN] judge_stats 获取失败，写入空结构以保证字段存在: {_e}")
        _tb.print_exc()
        return {
            "judge_calls": 0,
            "judge_sufficient": 0,
            "judge_sufficient_at_depth": [],
            "no_stop": None,
            "_error": f"{type(_e).__name__}: {_e}",
        }


def _get_sbert():
    global _sbert_model
    if _sbert_model is not None:
        return _sbert_model
    try:
        from sentence_transformers import SentenceTransformer
        _sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("[PoG] SBERT 模型加载成功")
    except Exception:
        _sbert_model = False
        print("[PoG] SBERT 不可用，Fuzzy Selection 使用关键词匹配兜底")
    return _sbert_model


# ================================================================
# 数据结构
# ================================================================
class Path:
    """一条从 topic entity 出发的推理路径"""
    __slots__ = ("nodes", "relations", "directions", "score")

    def __init__(self, start_entity: str):
        self.nodes = [start_entity]
        self.relations = []
        self.directions = []
        self.score = 0.0

    def extend(self, relation: str, entity: str, direction: str) -> "Path":
        p = Path.__new__(Path)
        p.nodes = self.nodes + [entity]
        p.relations = self.relations + [relation]
        p.directions = self.directions + [direction]
        p.score = 0.0
        return p

    @property
    def head(self) -> str:
        return self.nodes[0]

    @property
    def tail(self) -> str:
        return self.nodes[-1]

    @property
    def length(self) -> int:
        return len(self.relations)

    def to_text(self, names: Dict[str, str]) -> str:
        parts = []
        for i, node in enumerate(self.nodes):
            parts.append(f"{{{names.get(node, node)}}}")
            if i < len(self.relations):
                rel_short = self.relations[i].split("/")[-1] if "/" in self.relations[i] else self.relations[i]
                arrow = " → " if self.directions[i] == "tail" else " ← "
                parts.append(f"{arrow}{rel_short}{arrow}")
        return "".join(parts)

    def __repr__(self):
        return f"Path({' -> '.join(self.nodes)})"


# ================================================================
# 辅助函数
# ================================================================
def is_answer_match(name: str, entity_id: str, answer_set: Set[str]) -> bool:
    if not answer_set:
        return False
    n = str(name).strip().lower()
    e = str(entity_id).strip().lower()
    for ans in answer_set:
        a = str(ans).strip().lower()
        if not a:
            continue
        if a == n or a == e:
            return True
        if a in n or n in a:
            return True
        if a in e or e in a:
            return True
    return False


def _rel_short(rel: str) -> str:
    return rel.split("/")[-1] if "/" in rel else rel


def _build_graph_dict(paths: List[Path], names: Dict[str, str]) -> Dict:
    """将路径集合转换为 adjacency graph dict（供 run_experiment 统计用）"""
    graph = {}
    for p in paths:
        for i in range(len(p.relations)):
            head = p.nodes[i]
            tail = p.nodes[i + 1]
            if head not in graph:
                graph[head] = {}
            if tail not in graph[head]:
                graph[head][tail] = {"forward": set()}
            graph[head][tail]["forward"].add(p.relations[i])
    return graph


# ================================================================
# Phase 1: Initialization — Subgraph Detection
# 注：论文中 Entity Linking（LLM抽取候选实体→SBERT对齐KG→阈值判断→重试）
# 在本工程中简化：实验数据集直接提供 Freebase entity ID（CWQ/MetaQA 标准做法），
# 因此跳过 Entity Linking 步骤，直接从给定 topic entities 构建子图。
# ================================================================
def _build_subgraph(entity_ids: List[str], max_depth: int, names: Dict[str, str],
                    question: str = "", run_llm_func=None, planning=None):
    """
    从 topic entities 出发，提取 D_max 跳邻域子图。
    返回 adjacency dict: subgraph[entity] = list[dict{relation, connectedEntity, ...}]
    
    【改进插入点】如果 enable_relation_filter=True，每跳展开前对候选关系做 Top-K 筛选。
    """
    subgraph = {}
    visited = set()
    frontier = set(entity_ids)

    for depth in range(1, max_depth + 1):
        next_frontier = set()
        for eid in frontier:
            if eid in visited:
                continue
            visited.add(eid)

            neighbors = get_1hop_neighbors(eid)
            if not neighbors:
                subgraph[eid] = []
                continue

            # ===== B/D 组改进：关系筛选 Top-K（子图构建用快速关键词模式） =====
            if _IMPROVEMENTS_OK and _pog_get_config("enable_relation_filter"):
                all_rels = list({n["relation"] for n in neighbors})
                filtered_rels = filter_relations_topk(
                    all_rels, question, run_llm_func=run_llm_func,
                    force_fast=True  # 子图构建阶段用关键词筛选，避免大量LLM调用
                )
                filtered_set = set(filtered_rels)
                neighbors = [n for n in neighbors if n["relation"] in filtered_set]

            subgraph[eid] = neighbors
            for n in neighbors:
                cid = n["connectedEntity"]
                cname = n["connectedEntityName"]
                if cid not in names:
                    names[cid] = cname
                if cid not in visited:
                    next_frontier.add(cid)

        frontier = next_frontier
        if not frontier:
            break

    return subgraph


# ================================================================
# Phase 2a: Topic Entity Path Exploration
# ================================================================
def _entity_path_find(ordered_entities: List[str], depth: int,
                      subgraph: Dict, names: Dict[str, str]) -> List[Path]:
    """
    在预构建子图中搜索从 ordered_entities 出发、长度为 depth 的路径。
    对齐论文 EntityPathFind(List_T, D, G_q)。
    """
    if not ordered_entities:
        return []

    all_paths = []
    for start in ordered_entities:
        queue = [Path(start)]
        for d in range(depth):
            next_queue = []
            for path in queue:
                tail = path.tail
                neighbors = subgraph.get(tail, [])
                for n in neighbors:
                    cid = n["connectedEntity"]
                    if cid in path.nodes:
                        continue
                    new_path = path.extend(n["relation"], cid, n["direction"])
                    next_queue.append(new_path)
            queue = next_queue
            if not queue:
                break
        all_paths.extend(queue)

    return all_paths


# ================================================================
# Phase 3: Path Pruning — 三步 Beam Search
# ================================================================
def _step1_branch_reduced(paths: List[Path], topic_entities: Set[str]) -> List[Path]:
    """
    Step 1: Branch Reduced Selection
    移除不与任何其他 topic entity 连接的路径，优先保留连接多个 topic entities 的路径。
    """
    if len(topic_entities) <= 1:
        return paths

    scored = []
    for p in paths:
        nodes_set = set(p.nodes)
        connected = nodes_set & topic_entities
        score = len(connected)
        scored.append((score, p))
    scored.sort(key=lambda x: -x[0])
    return [p for _, p in scored]


def _step2_fuzzy_selection(paths: List[Path], indicator: str,
                           names: Dict[str, str], w1: int) -> List[Path]:
    """
    Step 2: Fuzzy Selection (SBERT 语义匹配)
    用 SBERT 将路径文本和 indicator 编码，按 cosine similarity 排序取 Top-W1。
    """
    if not paths or not indicator:
        return paths[:w1]

    sbert = _get_sbert()
    path_texts = [p.to_text(names) for p in paths]

    if sbert and sbert is not False:
        try:
            import numpy as np
            all_texts = [indicator] + path_texts
            embeddings = sbert.encode(all_texts, convert_to_numpy=True, show_progress_bar=False)
            indicator_emb = embeddings[0]
            path_embs = embeddings[1:]
            norms_p = np.linalg.norm(path_embs, axis=1, keepdims=True)
            norms_p[norms_p == 0] = 1.0
            norm_i = np.linalg.norm(indicator_emb)
            if norm_i == 0:
                norm_i = 1.0
            similarities = (path_embs @ indicator_emb) / (norms_p.flatten() * norm_i)
            ranked_idx = np.argsort(-similarities)
            selected = []
            for idx in ranked_idx[:w1]:
                paths[idx].score = float(similarities[idx])
                selected.append(paths[idx])
            return selected
        except Exception as e:
            print(f"[PoG] SBERT Fuzzy Selection 异常，回退关键词: {e}")

    # 兜底：关键词重叠打分
    indicator_tokens = set(re.findall(r"\w+", indicator.lower()))
    scored = []
    for i, p in enumerate(paths):
        ptokens = set(re.findall(r"\w+", path_texts[i].lower()))
        overlap = len(indicator_tokens & ptokens)
        p.score = overlap / max(len(indicator_tokens), 1)
        scored.append(p)
    scored.sort(key=lambda x: -x.score)
    return scored[:w1]


def _step3_precise_selection(paths: List[Path], question: str, indicator: str,
                             names: Dict[str, str], wmax: int,
                             run_llm_func=None, planning=None) -> List[Path]:
    """
    Step 3: Precise Path Selection (LLM)
    将 Top-W1 路径送入 LLM 精确打分，选 Top-Wmax。
    """
    if not paths:
        return []
    if len(paths) <= wmax:
        return paths
    if run_llm_func is None or FAST_MODE:
        return paths[:wmax]

    path_descs = []
    for i, p in enumerate(paths[:20]):
        path_descs.append(f"[{i+1}] {p.to_text(names)}")

    sub_q = ""
    if planning and planning.get("sub_questions"):
        sub_q = "\nSplit questions:\n" + "\n".join(f"- {sq}" for sq in planning["sub_questions"][:4])

    prompt = f"""Given the main question, an LLM-generated thinking chain (indicator), split questions, and candidate knowledge graph paths.
Select the top {wmax} most relevant paths that can help answer the question.

Question: {question}
Indicator: {indicator}
{sub_q}

Candidate Paths:
{chr(10).join(path_descs)}

Reply with ONLY a JSON array of path numbers (e.g., [1, 3, 5]). No explanation."""

    try:
        resp = run_llm_func(prompt, model="gpt4", temperature=0.1) or ""
        m = re.search(r"\[[\d,\s]+\]", resp)
        if m:
            selected_ids = json.loads(m.group(0))
            result = []
            for sid in selected_ids:
                idx = int(sid) - 1
                if 0 <= idx < len(paths):
                    result.append(paths[idx])
            if result:
                return result[:wmax]
    except Exception as e:
        print(f"[PoG] Precise Selection LLM 异常: {e}")

    return paths[:wmax]


# 路径数超阈值时二次剪枝的阈值
_PATH_REPRUNE_THRESHOLD = 50


def _path_pruning(paths: List[Path], question: str, indicator: str,
                  topic_entities: Set[str], names: Dict[str, str],
                  w1: int, wmax: int, run_llm_func=None, planning=None) -> List[Path]:
    """
    三步 Beam Search 路径剪枝（对齐论文 Phase 3 + 图5）
    Step 1: Branch Reduced Selection
    Step 2: Fuzzy Selection (SBERT cosine → Top-W1)
    Step 3: Precise Selection (LLM → Top-Wmax)
    图5附加：如果剪枝后路径数仍超过阈值，LLM 再次评估并剪枝。
    """
    if not paths:
        return []
    # Step 1: Branch Reduced
    paths = _step1_branch_reduced(paths, topic_entities)
    # Step 2: Fuzzy Selection (SBERT)
    paths = _step2_fuzzy_selection(paths, indicator, names, w1)
    # Step 3: Precise Selection (LLM)
    paths = _step3_precise_selection(paths, question, indicator, names, wmax,
                                     run_llm_func, planning)

    # 图5附加：路径数是否超过阈值 → 是 → LLM 再次评估并剪枝
    if len(paths) > _PATH_REPRUNE_THRESHOLD:
        print(f"[PoG] 路径数({len(paths)})超阈值({_PATH_REPRUNE_THRESHOLD})，再次剪枝")
        paths = _step3_precise_selection(paths, question, indicator, names, wmax,
                                         run_llm_func, planning)

    return paths


# ================================================================
# Phase 4: Question Answering Evaluation
# ================================================================
def _question_answering_eval(paths: List[Path], question: str, indicator: str,
                             names: Dict[str, str], run_llm_func=None,
                             planning=None) -> Tuple[bool, str]:
    """
    对齐论文 4.4 QA Evaluation：判断当前路径是否足以回答问题。
    返回 (sufficient, answer_text)。
    """
    if not paths:
        return False, ""
    if run_llm_func is None or FAST_MODE:
        return False, ""

    path_texts = [p.to_text(names) for p in paths[:5]]
    paths_section = "\n".join(f"- {t}" for t in path_texts)

    sub_q = ""
    if planning and planning.get("sub_questions"):
        sub_q = "\nSplit questions:\n" + "\n".join(f"- {sq}" for sq in planning["sub_questions"][:4])

    prompt = f"""Given the knowledge graph paths below, determine if they are sufficient to answer the given question.
If sufficient, respond with {{Yes}} followed by the answer.
If not sufficient, respond with {{No}} and explain what is missing.

Question: {question}
Indicator: {indicator}
{sub_q}

Related Paths:
{paths_section}

Your response:"""

    try:
        resp = run_llm_func(prompt, model="gpt4", temperature=0.1) or ""
        resp_clean = re.sub(r"<think>[\s\S]*?</think>", "", resp).strip()
        if "{Yes}" in resp_clean or resp_clean.lower().startswith("yes"):
            answer = re.sub(r"^\{?Yes\}?\s*[:\-,.]?\s*", "", resp_clean, flags=re.IGNORECASE).strip()
            if answer:
                return True, answer
        return False, ""
    except Exception as e:
        print(f"[PoG] QA Evaluation LLM 异常: {e}")
        return False, ""


# ================================================================
# Phase 2b: LLM Supplement Path Exploration
# ================================================================
def _supplement_prediction(paths: List[Path], question: str, indicator: str,
                           names: Dict[str, str], run_llm_func=None) -> List[Tuple[str, str]]:
    """
    对齐论文 Phase 2b：LLM 预测候选中间实体。
    返回 [(entity_name, supplement_indicator), ...]
    """
    if run_llm_func is None or FAST_MODE:
        return []

    path_texts = [p.to_text(names) for p in paths[:5]]
    paths_section = "\n".join(f"- {t}" for t in path_texts) if path_texts else "(no paths yet)"

    prompt = f"""Using the main question, the thinking chain, and related paths below, predict three intermediate entities that could help bridge the gap to the answer.

Question: {question}
Indicator: {indicator}

Related Paths:
{paths_section}

For each predicted entity, provide a chain of thought showing how it connects.
Reply as a JSON array of objects: [{{"entity": "...", "indicator": "..."}}]
No explanation, JSON only."""

    try:
        resp = run_llm_func(prompt, model="gpt4", temperature=0.3) or ""
        m = re.search(r"\[[\s\S]*?\]", resp)
        if m:
            items = json.loads(m.group(0))
            result = []
            for item in items[:3]:
                if isinstance(item, dict) and item.get("entity"):
                    result.append((str(item["entity"]).strip(),
                                   str(item.get("indicator", "")).strip()))
            return result
    except Exception as e:
        print(f"[PoG] Supplement Prediction 异常: {e}")
    return []


def _find_entity_in_subgraph(entity_name: str, subgraph: Dict,
                              names: Dict[str, str]) -> Optional[str]:
    """用 SBERT 或关键词匹配在子图中查找与预测实体名最接近的节点"""
    if not entity_name:
        return None

    best_id, best_score = None, 0.0
    target = entity_name.strip().lower()

    for eid, ename in names.items():
        n = str(ename).strip().lower()
        if n == target:
            return eid
        tokens_t = set(target.split())
        tokens_n = set(n.split())
        if tokens_t and tokens_n:
            overlap = len(tokens_t & tokens_n) / max(len(tokens_t | tokens_n), 1)
            if overlap > best_score:
                best_score = overlap
                best_id = eid

    if best_score >= 0.5:
        return best_id

    sbert = _get_sbert()
    if sbert and sbert is not False:
        try:
            import numpy as np
            candidates = list(names.items())[:500]
            texts = [entity_name] + [n for _, n in candidates]
            embs = sbert.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            sims = (embs[1:] @ embs[0]) / (
                np.linalg.norm(embs[1:], axis=1) * max(np.linalg.norm(embs[0]), 1e-9)
            )
            top_idx = int(np.argmax(sims))
            if sims[top_idx] > 0.6:
                return candidates[top_idx][0]
        except Exception:
            pass

    return None


# ================================================================
# Phase 2c: Node Expand Exploration（对齐图4：循环到 Dmax）
# ================================================================
def _node_expand(paths: List[Path], subgraph: Dict, names: Dict[str, str],
                 question: str = "", run_llm_func=None,
                 max_depth: int = 3) -> List[Path]:
    """
    对齐论文 Phase 2c + 图4：对路径末端未展开节点从源 KG 获取 1-hop 邻居，
    合并到已有路径形成新路径。循环扩展直到达到 Dmax 或无新路径。
    """
    all_new_paths = []
    current_paths = list(paths)

    for expand_round in range(max_depth):
        new_paths = []
        expanded = set()

        for p in current_paths:
            tail = p.tail
            if tail in expanded:
                continue
            if tail in subgraph:
                continue
            expanded.add(tail)

            neighbors = get_1hop_neighbors(tail)
            if not neighbors:
                continue

            # 改进插入点：关系筛选
            if _IMPROVEMENTS_OK and _pog_get_config("enable_relation_filter"):
                all_rels = list({n["relation"] for n in neighbors})
                filtered_rels = filter_relations_topk(
                    all_rels, question, run_llm_func=run_llm_func
                )
                filtered_set = set(filtered_rels)
                neighbors = [n for n in neighbors if n["relation"] in filtered_set]

            subgraph[tail] = neighbors
            for n in neighbors:
                cid = n["connectedEntity"]
                if cid not in names:
                    names[cid] = n["connectedEntityName"]

        for p in current_paths:
            tail = p.tail
            for n in subgraph.get(tail, []):
                cid = n["connectedEntity"]
                if cid not in set(p.nodes):
                    new_paths.append(p.extend(n["relation"], cid, n["direction"]))

        if not new_paths:
            print(f"[PoG]   Node Expand 第{expand_round+1}轮：无新路径，停止扩展")
            break

        print(f"[PoG]   Node Expand 第{expand_round+1}轮：发现 {len(new_paths)} 条新路径")
        all_new_paths.extend(new_paths)
        current_paths = new_paths

        # 图4：是否达到 Dmax → 是 → 停止
        max_path_len = max(p.length for p in new_paths) if new_paths else 0
        if max_path_len >= max_depth:
            print(f"[PoG]   Node Expand：路径长度达到 Dmax({max_depth})，停止")
            break

    return all_new_paths


# ================================================================
# 图4 左侧分支：LLM 伪路径补充（路径不充分时根据固有知识生成）
# ================================================================
def _llm_pseudo_path_supplement(question: str, indicator: str,
                                 existing_paths: List[Path],
                                 names: Dict[str, str],
                                 run_llm_func=None) -> List[Path]:
    """
    对齐图4左侧分支：当路径不充分时，LLM根据固有知识生成伪路径补充。
    返回伪路径（仅用于辅助QA，不修改子图）。
    """
    if run_llm_func is None or FAST_MODE:
        return []

    path_texts = [p.to_text(names) for p in existing_paths[:5]]
    paths_section = "\n".join(f"- {t}" for t in path_texts) if path_texts else "(no paths)"

    prompt = f"""The following KG reasoning paths are insufficient to answer the question.
Using your inherent knowledge, generate 2-3 supplementary reasoning chains that could bridge the gap.

Question: {question}
Indicator: {indicator}

Existing paths:
{paths_section}

For each supplementary chain, output one line in format:
entity1 --[relation]--> entity2 --[relation]--> entity3

Output only chains, no explanation."""

    try:
        resp = run_llm_func(prompt, model="gpt4", temperature=0.3) or ""
        pseudo_paths = []
        for ln in resp.strip().splitlines():
            s = ln.strip().lstrip("-").strip()
            if "--" in s and "-->" in s:
                # 创建一个伪路径对象，起点用占位符
                p = Path("pseudo_start")
                p.score = 0.1
                # 存入 relations 用于后续展示
                p.relations = [s]  # 存整行作为关系描述
                p.nodes = ["pseudo_start", "pseudo_end"]
                p.directions = ["tail"]
                pseudo_paths.append(p)
        print(f"[PoG] LLM伪路径补充：生成 {len(pseudo_paths)} 条")
        return pseudo_paths[:3]
    except Exception as e:
        print(f"[PoG] LLM伪路径补充异常: {e}")
        return []


# ================================================================
# 主入口
# ================================================================
def explore_graph_from_entities_by_hop_neighbor_1(
    entity_ids: List[str],
    max_depth: int = 3,
    answer_name: Optional[List[str]] = None,
    question: str = "",
    require_connect_for_success: bool = False,
    run_llm_func=None,
    ordered_entities: Optional[List[str]] = None,
    min_success_depth: int = 1,
    planning: Optional[Dict] = None,
    **kwargs,
):
    """
    PoG 完整流程主入口。

    Returns:
        tuple: (found, graph_dict, hop, all_entities, frontier, entity_names,
                early_stop, meta_dict)
    """
    t_start = time.time()
    print(f"\n[PoG] ========== 开始探索 ==========")
    print(f"[PoG] Topic entities: {entity_ids}")
    print(f"[PoG] Max depth: {max_depth}")

    ans_set = set(answer_name or [])
    if ordered_entities:
        entity_list = list(ordered_entities)
    else:
        entity_list = list(entity_ids)
    topic_set = set(entity_list)

    names: Dict[str, str] = {}
    for eid in entity_list:
        names[eid] = get_name(eid)
    print(f"[PoG] Topic 实体名: {', '.join(names[e] for e in entity_list)}")

    indicator = ""
    plan = planning if isinstance(planning, dict) else {}
    indicator = plan.get("indicator_text") or plan.get("planning_text") or ""

    w1 = POG_W1_FUZZY
    wmax = POG_WMAX
    phase_stats = []
    all_collected_paths = []
    found = False
    early_stop = False
    final_hop = max_depth

    # ================================================================
    # Phase 1: Subgraph Detection
    # ================================================================
    print(f"\n[PoG] Phase 1: 构建 {max_depth} 跳子图...")
    t0 = time.time()
    subgraph = _build_subgraph(entity_list, max_depth, names, question,
                               run_llm_func, planning)
    subgraph_time = time.time() - t0
    total_neighbors = sum(len(v) for v in subgraph.values())
    print(f"[PoG] 子图构建完成: {len(subgraph)} 个已展开节点, {total_neighbors} 条边, "
          f"{len(names)} 个实体, 耗时 {subgraph_time:.2f}s")
    phase_stats.append({"phase": "subgraph", "nodes": len(subgraph),
                        "edges": total_neighbors, "elapsed": round(subgraph_time, 2)})

    # 检查子图中是否已包含答案
    for eid, ename in names.items():
        if is_answer_match(ename, eid, ans_set):
            found = True
            print(f"[PoG] 子图已包含答案实体: {ename} ({eid})")
            break

    # ================================================================
    # Phase 2a: Topic Entity Path Exploration
    # ================================================================
    print(f"\n[PoG] Phase 2a: Topic Entity 路径探索...")
    d_start = min(min_success_depth, max_depth)
    topic_paths = []

    for D in range(d_start, max_depth + 1):
        t0 = time.time()
        paths_d = _entity_path_find(entity_list, D, subgraph, names)
        elapsed = time.time() - t0
        print(f"[PoG]   深度 {D}: 发现 {len(paths_d)} 条候选路径 ({elapsed:.2f}s)")

        if paths_d:
            pruned = _path_pruning(paths_d, question, indicator, topic_set, names,
                                   w1, wmax, run_llm_func, planning)
            print(f"[PoG]   深度 {D}: 剪枝后 {len(pruned)} 条路径")
            topic_paths = pruned
            all_collected_paths.extend(pruned)

            # QA Evaluation
            sufficient, answer = _question_answering_eval(
                pruned, question, indicator, names, run_llm_func, planning
            )
            phase_stats.append({"phase": "topic_path", "depth": D,
                                "raw_paths": len(paths_d), "pruned_paths": len(pruned),
                                "sufficient": sufficient, "elapsed": round(elapsed, 2)})
            if sufficient:
                print(f"[PoG]   深度 {D}: QA 评估 → 充足! 答案: {answer[:80]}")
                found = True
                final_hop = D
                break

            # ===== C/D 组改进：知识充足性判断（动态早停，对齐图10） =====
            if _IMPROVEMENTS_OK and _pog_get_config("enable_knowledge_judge"):
                # 图10：将当前路径集中的三元组转为自然语言描述
                pruned_texts = [p.to_text(names) for p in pruned]
                evidence_count = sum(p.length for p in pruned)
                rel_types = len({r for p in pruned for r in p.relations})
                is_sufficient = judge_knowledge_sufficiency(
                    paths_text=pruned_texts,
                    evidence_count=evidence_count,
                    rel_type_count=rel_types,
                    current_depth=D,
                    question=question,
                    run_llm_func=run_llm_func,
                )
                if is_sufficient:
                    if _pog_get_config("no_stop"):
                        # [D-PRIME] 判断照常发生并已记入统计，但不停止
                        print(f"[PoG]   [D' 判断不停] 深度 {D}: 判定充足，"
                              f"但 no_stop=True，继续探索（不早停）")
                    else:
                        print(f"[PoG]   [改进-知识判断] 深度 {D}: 判断为充足，提前停止")
                        early_stop = True
                        final_hop = D
                        break
        else:
            phase_stats.append({"phase": "topic_path", "depth": D,
                                "raw_paths": 0, "pruned_paths": 0,
                                "sufficient": False, "elapsed": round(elapsed, 2)})

    # ================================================================
    # Phase 2b: LLM Supplement Path Exploration
    # ================================================================
    if not found and not early_stop:
        print(f"\n[PoG] Phase 2b: LLM 补充路径探索...")
        t0 = time.time()
        predictions = _supplement_prediction(
            topic_paths, question, indicator, names, run_llm_func
        )
        print(f"[PoG]   LLM 预测 {len(predictions)} 个候选中间实体")

        for pred_name, pred_indicator in predictions:
            matched_eid = _find_entity_in_subgraph(pred_name, subgraph, names)
            if not matched_eid:
                print(f"[PoG]   候选 '{pred_name}' 未在子图中找到，跳过")
                continue
            print(f"[PoG]   候选 '{pred_name}' → {matched_eid}")

            sup_list = entity_list + [matched_eid]
            sup_indicator = pred_indicator or indicator
            sup_paths = _entity_path_find(sup_list, max_depth, subgraph, names)
            if sup_paths:
                pruned = _path_pruning(sup_paths, question, sup_indicator,
                                       topic_set | {matched_eid}, names,
                                       w1, wmax, run_llm_func, planning)
                all_collected_paths.extend(pruned)
                sufficient, answer = _question_answering_eval(
                    pruned, question, sup_indicator, names, run_llm_func, planning
                )
                if sufficient:
                    print(f"[PoG]   Supplement → 充足! 答案: {answer[:80]}")
                    found = True
                    break

        sup_elapsed = time.time() - t0
        phase_stats.append({"phase": "supplement", "predictions": len(predictions),
                            "elapsed": round(sup_elapsed, 2)})

    # ================================================================
    # Phase 2c: Node Expand Exploration（对齐图4：循环到 Dmax）
    # ================================================================
    if not found and not early_stop:
        print(f"\n[PoG] Phase 2c: 节点扩展探索（循环到Dmax）...")
        t0 = time.time()
        expand_paths = _node_expand(topic_paths or all_collected_paths,
                                    subgraph, names, question, run_llm_func,
                                    max_depth=max_depth)
        print(f"[PoG]   节点扩展共发现 {len(expand_paths)} 条新路径")

        if expand_paths:
            pruned = _path_pruning(expand_paths, question, indicator, topic_set, names,
                                   w1, wmax, run_llm_func, planning)
            all_collected_paths.extend(pruned)
            sufficient, answer = _question_answering_eval(
                pruned, question, indicator, names, run_llm_func, planning
            )
            if sufficient:
                print(f"[PoG]   Node Expand → 充足!")
                found = True

        expand_elapsed = time.time() - t0
        phase_stats.append({"phase": "node_expand", "new_paths": len(expand_paths),
                            "elapsed": round(expand_elapsed, 2)})

    # ================================================================
    # 图4 左侧分支：LLM 伪路径补充
    # ================================================================
    if not found and not early_stop:
        print(f"\n[PoG] 图4分支: LLM伪路径补充...")
        pseudo = _llm_pseudo_path_supplement(
            question, indicator, all_collected_paths, names, run_llm_func
        )
        if pseudo:
            all_collected_paths.extend(pseudo)

    # ================================================================
    # 汇总返回
    # ================================================================
    # 从路径中检查答案命中
    if not found:
        for p in all_collected_paths:
            for node in p.nodes:
                if is_answer_match(names.get(node, node), node, ans_set):
                    found = True
                    break
            if found:
                break

    graph_dict = _build_graph_dict(all_collected_paths, names)
    all_entities = set(names.keys())

    # 构建 top_paths_text 供 QA chain 使用
    top_paths_text = []
    seen_path_str = set()
    for p in all_collected_paths:
        txt = p.to_text(names)
        if txt not in seen_path_str:
            seen_path_str.add(txt)
            top_paths_text.append((txt, p.score))
    top_paths_text.sort(key=lambda x: -x[1])
    top_paths_text = top_paths_text[:10]

    total_elapsed = time.time() - t_start
    print(f"\n[PoG] ========== 探索完毕 ==========")
    print(f"[PoG] 答案命中: {'Y' if found else 'N'}  早停: {'Y' if early_stop else 'N'}  "
          f"路径: {len(all_collected_paths)}  实体: {len(all_entities)}  "
          f"总耗时: {total_elapsed:.2f}s")

    meta = {
        "phase_stats": phase_stats,
        "top_paths_text": top_paths_text,
    }
    # [D-PRIME] 带出本题的判断统计（供 run_experiment 记录）
    # [FIX-9] 原实现异常时静默 pass，会导致结果 JSON 静默缺少 judge_stats 字段，
    #         而 D/D' 对比分析完全依赖该字段 —— 失败必须可见。改走 _collect_judge_stats()。
    meta["judge_stats"] = _collect_judge_stats()

    return (found, graph_dict, final_hop, all_entities, set(), names,
            early_stop, meta)
