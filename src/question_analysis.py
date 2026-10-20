# -*- coding: utf-8 -*-
"""
Question Analysis (PoG Planning)
输出结构化 planning：
- topic_entities_ordered: 按问题文本出现顺序排列
- sub_questions: 轻量分解（对应论文 split questions）
- indicator_text: PoG 论文风格 LLM indicator（规则或 LLM 生成）
- d_predict: 预测推理深度
"""

import re
import os
import json
from typing import List, Dict

try:
    from llm import run_LLM
except Exception:
    run_LLM = None


def _normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _estimate_depth(question: str) -> int:
    ql = _normalize_text(question)
    hard_markers = ["before", "after", "earliest", "latest", "first", "last", "less than", "greater than"]
    bridge_markers = ["that", "which", "whose", "where", "who", "what", "when"]
    hard_cnt = sum(1 for m in hard_markers if m in ql)
    bridge_cnt = sum(1 for m in bridge_markers if m in ql)
    if hard_cnt >= 2 or bridge_cnt >= 2:
        return 3
    if hard_cnt >= 1 or bridge_cnt >= 1:
        return 2
    return 1


def _split_sub_questions(question: str) -> List[str]:
    if not question:
        return []
    parts = re.split(r"\b(and|then|after|before|which|whose)\b", question, flags=re.IGNORECASE)
    chunks = []
    buf = ""
    for p in parts:
        if p.strip().lower() in {"and", "then", "after", "before", "which", "whose"}:
            if buf.strip():
                chunks.append(buf.strip())
            buf = ""
        else:
            buf += " " + p
    if buf.strip():
        chunks.append(buf.strip())
    if not chunks:
        return [question.strip()]
    return chunks[:4]


def _order_topic_entities_by_mention(question: str, topic_entity_dict: Dict[str, str]) -> List[str]:
    if not topic_entity_dict:
        return []
    ql = _normalize_text(question)
    scored = []
    for eid, name in topic_entity_dict.items():
        n = _normalize_text(name)
        idx = ql.find(n) if n else -1
        scored.append((idx if idx >= 0 else 10**9, eid))
    scored.sort(key=lambda x: x[0])
    ordered = [eid for _, eid in scored]
    return ordered


def _rule_indicator(question: str, ordered_ids: List[str], topic_entity_dict: Dict[str, str]) -> str:
    """PoG 论文风格的 LLM indicator：用引号包裹的 topic 实体链 + 问题摘要（轻量规则版）。"""
    names = []
    for eid in ordered_ids:
        nm = str(topic_entity_dict.get(eid, "") or "").strip() or eid
        names.append(nm)
    if not names:
        return ""
    quoted = [f'"{n}"' for n in names]
    chain = " - ".join(quoted)
    stem = re.sub(r"\s+", " ", (question or "").strip())[:160]
    if stem:
        return f"{chain} — toward: {stem}"
    return chain


def _llm_planning(question: str, topic_entity_dict: Dict[str, str]) -> Dict:
    if run_LLM is None:
        return {}
    topic_json = json.dumps(topic_entity_dict, ensure_ascii=False)
    prompt = f"""You are a planner for KG multi-hop reasoning (PoG-style).
Question: {question}
Topic entities (id->name): {topic_json}
Return strict JSON with keys:
- d_predict (integer 1, 2, or 3): predicted max relational hops needed from topics to answer
- sub_questions (array of short strings): split questions, one per topic entity connection to the answer
- topic_entities_ordered (array of entity ids): same order as they should appear in reasoning
- indicator_text (string): one line, PoG-style chain like \\"EntityA\\" - relation hint - \\"EntityB\\" - ... - answer slot
No markdown, JSON only.
"""
    try:
        resp = run_LLM(prompt, model="gpt4", temperature=0.1) or ""
        m = re.search(r"\{[\s\S]*\}", resp)
        raw = m.group(0) if m else resp
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            return {}
        return obj
    except Exception:
        return {}


def _ensure_topic_entity_dict(topic_entity_raw) -> Dict[str, str]:
    """统一 topic_entity 格式，list/dict 均转为 dict。"""
    if isinstance(topic_entity_raw, dict):
        return {str(k): str(v) for k, v in topic_entity_raw.items()}
    if isinstance(topic_entity_raw, list):
        out = {}
        for item in topic_entity_raw:
            if isinstance(item, str) and item.strip():
                eid = item.strip()
                out[eid] = eid
            elif isinstance(item, dict):
                eid = str(item.get("id") or item.get("entity") or item.get("value") or "").strip()
                name = str(item.get("name") or item.get("label") or eid).strip()
                if eid:
                    out[eid] = name or eid
        return out
    return {}


def analyze_question(question: str, topic_entity_raw) -> Dict:
    """输出结构化 planning，用于PoG主流程"""
    topic_entity_dict = _ensure_topic_entity_dict(topic_entity_raw)
    use_llm = os.getenv("POG_USE_LLM_PLANNING", "0").strip() == "1"

    ordered = _order_topic_entities_by_mention(question, topic_entity_dict)
    base = {
        "d_predict": _estimate_depth(question),
        "sub_questions": _split_sub_questions(question),
        "topic_entities_ordered": ordered,
        "indicator_text": _rule_indicator(question, ordered, topic_entity_dict),
        "planning_text": "",
        "planning_source": "rule",
    }

    if use_llm:
        llm_plan = _llm_planning(question, topic_entity_dict)
        if llm_plan:
            dp = llm_plan.get("d_predict", base["d_predict"])
            try:
                dp = int(dp)
            except Exception:
                dp = base["d_predict"]
            if dp < 1:
                dp = 1
            if dp > 3:
                dp = 3
            sq = llm_plan.get("sub_questions", base["sub_questions"])
            if not isinstance(sq, list):
                sq = base["sub_questions"]
            te = llm_plan.get("topic_entities_ordered", base["topic_entities_ordered"])
            if not isinstance(te, list) or not te:
                te = base["topic_entities_ordered"]
            ind = llm_plan.get("indicator_text", "")
            if not isinstance(ind, str) or not ind.strip():
                ind = _rule_indicator(question, te, topic_entity_dict)
            else:
                ind = ind.strip()
            base.update({
                "d_predict": dp,
                "sub_questions": [str(x).strip() for x in sq if str(x).strip()][:6] or base["sub_questions"],
                "topic_entities_ordered": te,
                "indicator_text": ind,
                "planning_source": "llm",
            })

    if base["sub_questions"]:
        base["planning_text"] = " -> ".join(base["sub_questions"])
    elif base.get("indicator_text"):
        base["planning_text"] = base["indicator_text"]
    return base


def planning_for_explore(planning: Dict) -> Dict:
    """传给 pog_core 的 planning 子集（可序列化、字段稳定）。"""
    if not isinstance(planning, dict):
        return {}
    return {
        "indicator_text": planning.get("indicator_text") or planning.get("planning_text") or "",
        "sub_questions": planning.get("sub_questions") or [],
        "planning_text": planning.get("planning_text") or planning.get("indicator_text") or "",
    }
