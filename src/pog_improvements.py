# -*- coding: utf-8 -*-
"""
PoG 之上的可插拔改进（非 PoG 主干本身）

主干见 pog_core：Tan 等 PoG（WWW 2025）。本模块为蒋献等 LLMKG（《电信科学》2025，
doi:10.11959/j.issn.1000-0801.2025238）中的可插拔机制在本工程中的实现：
  1. 关系筛选 Top-K（Prompt 或 FAST_MODE 下关键词兜底）
  2. 知识充足性判断 / 动态早停（对应 LLMKG「知识是否足够—是否继续检索」）

每组改进均可：开/关；在开启时选用 LLM 或 FAST_MODE 下的规则兜底。
"""

import re
from typing import List, Dict, Set, Tuple

# ========================================
# 全局配置
# ========================================
CONFIG = {
    # 关系筛选
    "enable_relation_filter": True,
    "use_llm_filter":         True,
    "relation_filter_k":      10,
    "relation_filter_threshold": 15,  # 关系数≤此值时跳过筛选，直接全保留
    # 知识判断（与 config.PAPER_* 粗扫推荐一致，run_experiment 会再次注入）
    "enable_knowledge_judge": True,
    "use_llm_judge":          True,
    "min_evidence_threshold": 8,
    "max_evidence_threshold": 50,
    "min_relation_types":     4,
    "min_depth_before_stop":  2,
    # Prompt截断
    "max_relations_display":  50,
    "max_triples_display":    20,
}


def get_config(key: str):
    return CONFIG.get(key)


def set_config(key: str, value):
    CONFIG[key] = value
    print(f"[配置] {key} = {value}")


# ========================================
# 改进1：关系筛选 Top-K
# ========================================
def filter_relations_topk(
    all_relations: List[str],
    question: str,
    K: int = None,
    model: str = "gpt4",
    run_llm_func=None,
    use_fallback: bool = True,
    current_depth: int = 1,
    force_fast: bool = False,
) -> List[str]:
    """
    从候选关系中筛选Top-K个与问题最相关的关系
    force_fast=True时强制用关键词兜底（用于子图构建阶段，避免大量LLM调用）
    返回筛选后的关系列表
    """
    if not get_config("enable_relation_filter"):
        return all_relations

    if K is None:
        K = get_config("relation_filter_k")

    if len(all_relations) <= K:
        return all_relations

    # 阈值判断：关系数未超过阈值时不筛选
    threshold = get_config("relation_filter_threshold")
    if threshold and len(all_relations) <= threshold:
        return all_relations

    # 强制快速模式（子图构建阶段）或配置不使用LLM
    if force_fast or not get_config("use_llm_filter"):
        return _fallback_relation_filter(question, all_relations, K)

    # LLM模式
    max_rel = get_config("max_relations_display")
    candidates = all_relations[:max_rel]
    relations_str = "\n".join([f"- {r}" for r in candidates])
    prompt = f"""You are selecting KG relations for multi-hop QA.
Question: {question}
Current hop: {current_depth}

Candidate relations:
{relations_str}

Select the top {K} relations most likely to advance toward the final answer.
Principles:
1. Prefer relations that directly carry question-required semantics.
2. Prefer relations chainable in next hops to reach answer entities.
3. Avoid generic/weakly-informative relations (type/object/name/metadata).
4. Keep diversity: no near-duplicate relations.

Output ONLY relation ids, one per line, no explanation."""

    try:
        if run_llm_func is None:
            raise ValueError("run_llm_func为None")
        print(f"[关系筛选] LLM模式：筛选Top-{K}...")
        response = run_llm_func(prompt, model, temperature=0.2)
        if not response:
            raise ValueError("LLM返回为空")
        selected = _parse_relation_output(response, candidates)
        valid = [r for r in selected if r in candidates]
        if valid:
            result = valid[:K]
            print(f"[关系筛选] LLM筛选完成：{len(candidates)} -> {len(result)}")
            return result
        raise ValueError("LLM输出无有效关系")
    except Exception as e:
        print(f"[关系筛选] LLM失败({e})，使用兜底规则")
        if use_fallback:
            return _fallback_relation_filter(question, working, K)
        return working[:K]


def _parse_relation_output(response: str, valid_relations: List[str]) -> List[str]:
    """解析LLM输出的关系列表"""
    response = response.strip()
    parts = []
    if ',' in response:
        parts = response.split(',')
    elif '\n' in response:
        parts = response.split('\n')
    else:
        parts = response.split()

    parsed = []
    for p in parts:
        cleaned = re.sub(r'^\d+[\.)\s]+', '', p.strip()).strip('"\'')
        if not cleaned:
            continue
        if cleaned in valid_relations:
            parsed.append(cleaned)
        else:
            for vr in valid_relations:
                if cleaned.lower() in vr.lower() or vr.lower() in cleaned.lower():
                    parsed.append(vr)
                    break
    return parsed


def _rank_relations_by_keywords(
    question: str,
    all_relations: List[str],
) -> List[str]:
    """按与问题的关键词相关性排序（高分在前）。"""
    normalized_blocks = [question.lower()] if question else []
    priority_rels = ['spouse', 'nationality', 'profession', 'birth', 'death',
                     'location', 'parent', 'child', 'education', 'award',
                     'country', 'language', 'religion', 'currency', 'film']
    scores = []
    for rel in all_relations:
        score = 0.0
        rl = rel.lower()
        rel_parts = [x for x in re.split(r"[._/]+", rl) if x]
        rel_tokens = set(rel_parts)
        for idx, block in enumerate(normalized_blocks):
            weight = 1.0 if idx == 0 else 0.8
            if rl in block:
                score += 10 * weight
            for word in block.split():
                if len(word) <= 2:
                    continue
                if word in rl:
                    score += 5 * weight
                if word in rel_tokens:
                    score += 4 * weight
            if any(part in block for part in rel_parts if len(part) > 3):
                score += 2 * weight
        if any(pr in rl for pr in priority_rels):
            score += 3
        scores.append((rel, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [r for r, _ in scores]


def _fallback_relation_filter(question: str, all_relations: List[str], K: int) -> List[str]:
    """兜底规则：基于关键词匹配打分"""
    ranked = _rank_relations_by_keywords(question, all_relations)
    result = ranked[:K]
    print(f"[关系筛选] 兜底规则完成：{len(all_relations)} -> {len(result)}")
    return result


# ========================================
# 改进2：知识充足性判断（动态早停）
# ========================================
def judge_knowledge_sufficiency(
    paths_text: List[str] = None,
    evidence_count: int = 0,
    rel_type_count: int = 0,
    current_depth: int = 1,
    question: str = "",
    model: str = "gpt4",
    run_llm_func=None,
    use_fallback: bool = True,
) -> bool:
    """
    判断当前证据是否足以回答问题（对齐图10流程）。
    paths_text: 当前剪枝后路径集转为的自然语言描述列表。
    返回 True=证据充足可早停，False=继续扩展
    """
    if not get_config("enable_knowledge_judge"):
        return False

    if evidence_count <= 0 and not paths_text:
        return False

    min_depth = get_config("min_depth_before_stop")
    if current_depth < min_depth:
        print(f"[知识判断] 深度({current_depth}) < 最小深度({min_depth})，继续扩展")
        return False

    if not get_config("use_llm_judge"):
        print(f"[知识判断] 兜底规则模式")
        return _fallback_sufficiency_judge_simple(evidence_count, rel_type_count, current_depth)

    # LLM模式：将路径三元组转为自然语言描述传入（对齐图10）
    evidence_section = "(no paths)"
    if paths_text:
        shown = paths_text[:10]
        evidence_section = "\n".join([f"- {p}" for p in shown])
        if len(paths_text) > 10:
            evidence_section += f"\n... ({len(paths_text) - 10} more paths omitted)"

    prompt = f"""You are evaluating whether the current KG reasoning paths are sufficient to answer a question.

Question: {question}
Current hop: {current_depth}
Number of evidence triples: {evidence_count}
Number of relation types: {rel_type_count}

Current reasoning paths (natural language):
{evidence_section}

Decision rule:
- Return SUFFICIENT only if the provided paths already contain a complete and unambiguous reasoning chain to the final answer.
- Return INSUFFICIENT if key relation/entity is still missing, constraints are not satisfied, or multiple competing candidates remain.

Output exactly one word: SUFFICIENT or INSUFFICIENT"""

    try:
        if run_llm_func is None:
            raise ValueError("run_llm_func为None")
        print(f"[知识判断] LLM模式：判断证据充足性（{len(paths_text or [])}条路径）...")
        response = run_llm_func(prompt, model, temperature=0.1)
        if not response:
            raise ValueError("LLM返回为空")
        rl = response.strip().lower()
        if any(w in rl for w in ['insufficient', 'not sufficient', 'no', '不足', '否']):
            print(f"[知识判断] LLM判断：不足，继续扩展")
            return False
        if any(w in rl for w in ['sufficient', 'yes', '足够', '是']):
            print(f"[知识判断] LLM判断：充足，可早停")
            return True
        raise ValueError(f"LLM输出无法解析: {response}")
    except Exception as e:
        print(f"[知识判断] LLM失败({e})，使用兜底规则")
        if use_fallback:
            return _fallback_sufficiency_judge_simple(evidence_count, rel_type_count, current_depth)
        return False


# ========================================
# PoG论文：Path Summarizing（降低噪声 + 帮助评估/回答）
# ========================================
def summarize_paths_for_qa(
    question: str,
    planning_text: str,
    sub_questions: List[str],
    top_paths_text: List[str],
    model: str = "gpt4",
    run_llm_func=None,
) -> str:
    """
    对齐PoG论文的 Path Summarizing prompt：把Top paths总结成更短、更“可用”的证据链/三元组。
    输出尽量短的 facts 列表文本（不给解释）。
    """
    if run_llm_func is None:
        return ""
    if not question or not top_paths_text:
        return ""
    sq_lines = "\n".join([f"- {x}" for x in (sub_questions or [])[:4]]) or "(none)"
    path_lines = "\n".join([f"- {p}" for p in top_paths_text[:8]]) or "(none)"
    plan = planning_text or ""
    prompt = f"""Given a main question, an uncertain thinking indicator, and retrieved KG reasoning paths, summarize the key knowledge triples needed for answering.

Main question: {question}
Thinking indicator: {plan}
Split questions:
{sq_lines}
Related paths:
{path_lines}

Rules:
- Only use entities/relations that appear in the provided paths.
- Do NOT add new entities or relations.
- Output a compact list of factual triples in plain text, one per line, like:
  head --[relation]--> tail
- Output only triples, no explanation.
"""
    try:
        resp = run_llm_func(prompt, model=model, temperature=0.0) or ""
        resp = resp.strip()
        # 轻清洗：只保留看起来像 triple 的行
        lines = []
        for ln in resp.splitlines():
            s = ln.strip().lstrip("-").strip()
            if not s:
                continue
            if "--" in s and "-->" in s and "[" in s and "]" in s:
                lines.append(s)
        return "\n".join(lines[:30]).strip()
    except Exception:
        return ""


# ========================================
# PoG 论文 Appendix E：问答评估 + 答案生成（两步）
# ========================================
def qa_evaluate_paths_paper_style(
    question: str,
    think_indicator: str,
    sub_questions: List[str],
    related_paths: str,
    model: str = "gpt4",
    run_llm_func=None,
) -> Tuple[bool, str]:
    """
    Question Answering Evaluation：先判 split questions 再判主问是否可由当前路径充分回答。
    论文约定：充分则 {Yes} 并给出主问答案；否则 {No} 并简述原因。
    返回 (sufficient, raw_llm_text)。
    """
    if run_llm_func is None:
        return False, ""
    think = (think_indicator or "").strip() or "(uncertain chain-of-thought)"
    sq = sub_questions or []
    split_block = "\n".join([f"split_question {i+1}: {x}" for i, x in enumerate(sq[:6])]) or "(none)"
    rp = (related_paths or "").strip() or "(no paths)"
    prompt = f"""Given a main question, an uncertain LLM-generated thinking indicator that considers the topic entities, a few split questions, and retrieved knowledge graph paths (entities with ids like m/...) -> relationships -> entities.

Main question: {question}
Think indicator: {think}
{split_block}

Related paths (summarized or structured):
{rp}

Your task: determine whether these paths are sufficient to answer the split questions first, then the main question.

Output format (strict):
- If sufficient: first line must contain {{Yes}}, then a line starting with "Answer:" followed ONLY by the main-question answer (entity name or short value from the path when possible).
- If not sufficient: first line must contain {{No}}, then briefly explain what is missing (one short sentence).

Do not output other sections."""
    try:
        resp = (run_llm_func(prompt, model=model, temperature=0.0) or "").strip()
        low = resp.lower()
        if "{no}" in low or re.search(r"\bno\b", low[:80]):
            return False, resp
        if "{yes}" in low or re.search(r"\byes\b", low[:120]):
            return True, resp
        return False, resp
    except Exception:
        return False, ""


def _extract_answer_after_yes(eval_text: str) -> str:
    """从评估回复中解析 Answer: 后的短语。"""
    if not eval_text:
        return ""
    m = re.search(r"answer\s*:\s*(.+)", eval_text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        line = m.group(1).strip().split("\n")[0].strip()
        line = line.strip("{}\"'")
        return line
    # 备用：最后一行非 Yes/No 的短行
    for ln in reversed(eval_text.strip().splitlines()):
        s = ln.strip()
        if not s or "{yes}" in s.lower() or "{no}" in s.lower():
            continue
        if s.lower().startswith("answer"):
            continue
        if len(s) < 80 and not s.lower().startswith("if "):
            return s.strip("{}\"'")
    return ""


def qa_generate_answer_paper_style(
    question: str,
    think_indicator: str,
    sub_questions: List[str],
    related_paths: str,
    model: str = "gpt4",
    run_llm_func=None,
) -> str:
    """Question Answering Generation：基于路径与自身知识生成最终答案（论文 Appendix E）。"""
    if run_llm_func is None:
        return ""
    think = (think_indicator or "").strip() or "(uncertain chain-of-thought)"
    sq = sub_questions or []
    split_block = "\n".join([f"split_question {i+1}: {x}" for i, x in enumerate(sq[:6])]) or "(none)"
    rp = (related_paths or "").strip() or "(no paths)"
    prompt = f"""Given a main question, thinking indicator, split questions, and retrieved knowledge graph paths, generate the final answer.

Main question: {question}
Think indicator: {think}
{split_block}

Related paths:
{rp}

Rules:
- Answer with ONLY the final answer (one entity name, number, or short phrase).
- Prefer facts stated in the paths; you may use minimal inherent knowledge only to connect obvious dots.
- Do NOT explain. Do NOT say you cannot answer."""

    try:
        resp = (run_llm_func(prompt, model=model, temperature=0.0) or "").strip()
        if "\n" in resp:
            resp = resp.split("\n")[0].strip()
        return resp.strip("\"'{}")
    except Exception:
        return ""


def run_pog_paper_answer_chain(
    question: str,
    planning: Dict,
    top_paths_scored: List[Tuple[str, float]],
    run_llm_func,
    model: str = "gpt4",
) -> str:
    """
    Summarize → Evaluate → (optional extract) → Generate
    planning: indicator_text / sub_questions / planning_text
    top_paths_scored: [(path_text, score), ...]
    """
    if run_llm_func is None:
        return ""
    plan = planning if isinstance(planning, dict) else {}
    think = plan.get("indicator_text") or plan.get("planning_text") or ""
    sub_q = plan.get("sub_questions") or []
    if not isinstance(sub_q, list):
        sub_q = []

    path_lines = [pt for pt, _sc in (top_paths_scored or []) if pt]
    if not path_lines:
        return ""

    summarized = summarize_paths_for_qa(
        question=question,
        planning_text=think,
        sub_questions=sub_q,
        top_paths_text=path_lines,
        model=model,
        run_llm_func=run_llm_func,
    )
    related = ""
    if summarized:
        related = "Summarized triples:\n" + summarized + "\n\nRaw paths:\n" + "\n".join([f"- {p}" for p in path_lines[:6]])
    else:
        related = "\n".join([f"- {p}" for p in path_lines[:12]])

    ok, eval_resp = qa_evaluate_paths_paper_style(
        question=question,
        think_indicator=think,
        sub_questions=sub_q,
        related_paths=related,
        model=model,
        run_llm_func=run_llm_func,
    )
    if ok:
        ans = _extract_answer_after_yes(eval_resp)
        if ans and len(ans.split()) <= 16:
            return ans

    return qa_generate_answer_paper_style(
        question=question,
        think_indicator=think,
        sub_questions=sub_q,
        related_paths=related,
        model=model,
        run_llm_func=run_llm_func,
    )


def _format_evidence(
    evidence_triples: Dict[Tuple[str, str], Set[str]],
    entity_names: Dict[str, str] = None,
    max_display: int = 20
) -> str:
    lines = []
    count = 0
    for (head, relation), tails in evidence_triples.items():
        if count >= max_display:
            lines.append(f"... (还有{len(evidence_triples) - count}条未显示)")
            break
        head_name = entity_names.get(head, head) if entity_names else head
        for tail in list(tails)[:5]:
            tail_name = entity_names.get(tail, tail) if entity_names else tail
            lines.append(f"- {head_name} --[{relation}]--> {tail_name}")
            count += 1
            if count >= max_display:
                break
    return "\n".join(lines)


def _fallback_sufficiency_judge_simple(
    evidence_count: int,
    rel_type_count: int,
    current_depth: int = 1
) -> bool:
    """兜底规则：基于证据数量和关系类型判断"""
    min_ev  = get_config("min_evidence_threshold")
    max_ev  = get_config("max_evidence_threshold")
    min_rel = get_config("min_relation_types")
    min_d   = get_config("min_depth_before_stop")

    if current_depth < min_d:
        return False
    if evidence_count < min_ev:
        print(f"[知识判断] 证据数({evidence_count}) < {min_ev}，不足")
        return False
    if evidence_count > max_ev:
        print(f"[知识判断] 证据数({evidence_count}) > {max_ev}，充足")
        return True
    if rel_type_count >= min_rel:
        print(f"[知识判断] 关系类型({rel_type_count}) >= {min_rel}，充足")
        return True
    if evidence_count >= min_ev * 2 and rel_type_count >= min_rel - 1:
        print(f"[知识判断] 综合判断充足")
        return True
    print(f"[知识判断] 默认不足，继续扩展 (证据={evidence_count}, 关系类型={rel_type_count})")
    return False

