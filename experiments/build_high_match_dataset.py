# NOTE: service address and credentials are read from environment
#       variables (LLM_BASE_URL / LLM_API_KEY / LLM_MODEL).
#       This file was sanitised before public release.
# -*- coding: utf-8 -*-
"""
高匹配样本构建脚本
功能：
1. 从 cwq.json 中抽样（默认与现有实验一致）
2. 基于局部知识图谱做启发式匹配度评估
3. 输出 high / medium / low 标签
4. 导出 high match 子集与统计报告
"""

import os
import json
from datetime import datetime
from collections import defaultdict

from config import DATASET_PATH, DATASET_FILE, SAMPLE_SIZE, SAMPLE_STRATEGY
from freebase_func import _get_rdflib_graph, id2entity_name_or_type

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "experiment_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FREEBASE_PREFIX = "http://freebase.com/"
NAME_PRED = FREEBASE_PREFIX + "type.object.name"
ALIAS_PRED = FREEBASE_PREFIX + "common.topic.alias"


def fix_entity_id(eid):
    if isinstance(eid, str) and eid.startswith("m.") and "/" not in eid:
        return "m/" + eid[2:]
    return eid


def select_balanced_samples(valid_data, sample_size):
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


def normalize_text(text):
    import re
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[-_/,:;]+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def uri_for_entity(entity_id):
    from rdflib import URIRef
    return URIRef(FREEBASE_PREFIX + entity_id)


def uri_local_name(uri_obj):
    value = str(uri_obj)
    if value.startswith(FREEBASE_PREFIX):
        return value.replace(FREEBASE_PREFIX, "")
    return value


def literal_to_text(obj):
    try:
        return str(obj)
    except Exception:
        return ""


def collect_entity_neighbors(graph, entity_id, max_first_hop=80, max_second_hop=120):
    from rdflib import URIRef, Literal

    entity_uri = uri_for_entity(entity_id)
    first_hop_entities = set()
    second_hop_entities = set()
    relation_set = set()
    text_values = set()
    out_edges = 0
    in_edges = 0

    for s, p, o in graph.triples((entity_uri, None, None)):
        out_edges += 1
        relation_set.add(uri_local_name(p))
        if isinstance(o, URIRef):
            first_hop_entities.add(uri_local_name(o))
        elif isinstance(o, Literal):
            text = literal_to_text(o)
            if text:
                text_values.add(text)
        if out_edges >= max_first_hop:
            break

    for s, p, o in graph.triples((None, None, entity_uri)):
        in_edges += 1
        relation_set.add(uri_local_name(p))
        if isinstance(s, URIRef):
            first_hop_entities.add(uri_local_name(s))
        if in_edges >= max_first_hop:
            break

    second_count = 0
    for nbr in list(first_hop_entities)[:30]:
        nbr_uri = uri_for_entity(nbr) if not nbr.startswith("http") else URIRef(nbr)
        for s, p, o in graph.triples((nbr_uri, None, None)):
            relation_set.add(uri_local_name(p))
            if isinstance(o, URIRef):
                second_hop_entities.add(uri_local_name(o))
            elif isinstance(o, Literal):
                text = literal_to_text(o)
                if text:
                    text_values.add(text)
            second_count += 1
            if second_count >= max_second_hop:
                break
        if second_count >= max_second_hop:
            break

    name_texts = set()
    for candidate in list(first_hop_entities)[:20] + list(second_hop_entities)[:20]:
        if candidate.startswith("m/") or candidate.startswith("g/"):
            try:
                name = id2entity_name_or_type(candidate)
                if name and name != "Unnamed Entity":
                    name_texts.add(name)
            except Exception:
                pass

    try:
        topic_name = id2entity_name_or_type(entity_id)
        if topic_name and topic_name != "Unnamed Entity":
            name_texts.add(topic_name)
    except Exception:
        pass

    return {
        "out_edges": out_edges,
        "in_edges": in_edges,
        "first_hop_entity_count": len(first_hop_entities),
        "second_hop_entity_count": len(second_hop_entities),
        "relation_count": len(relation_set),
        "texts": sorted(list(text_values | name_texts))[:120],
    }


def score_sample(graph, item):
    topic_entity = item.get("topic_entity", {}) or {}
    answer = item.get("answer", "")
    ctype = item.get("compositionality_type", "unknown")

    if not topic_entity:
        return {
            "match_label": "low",
            "score": 0,
            "reasons": ["无topic_entity"],
            "topic_metrics": [],
            "answer_visible": False,
            "answer": answer,
            "compositionality_type": ctype,
        }

    topic_metrics = []
    total_score = 0
    reasons = []
    collected_texts = set()

    for raw_eid, raw_name in topic_entity.items():
        eid = fix_entity_id(raw_eid)
        metrics = collect_entity_neighbors(graph, eid)
        metrics["entity_id"] = eid
        metrics["entity_name"] = raw_name
        topic_metrics.append(metrics)
        collected_texts.update(metrics["texts"])

        if metrics["out_edges"] + metrics["in_edges"] > 0:
            total_score += 2
            reasons.append(f"主题实体{eid}存在邻接边")
        else:
            reasons.append(f"主题实体{eid}邻域稀疏")

        if metrics["first_hop_entity_count"] >= 2:
            total_score += 1
        if metrics["second_hop_entity_count"] >= 2:
            total_score += 1
        if metrics["relation_count"] >= 3:
            total_score += 1

    answer_visible = False
    norm_answer = normalize_text(answer)
    if norm_answer:
        for text in collected_texts:
            norm_text = normalize_text(text)
            if norm_answer and (norm_answer in norm_text or norm_text in norm_answer):
                answer_visible = True
                break

    if answer_visible:
        total_score += 3
        reasons.append("答案在局部图谱邻域文本中可见")
    else:
        reasons.append("答案在局部图谱邻域中不可见")

    if ctype in {"comparative", "superlative"}:
        if any(any(k in normalize_text(t) for k in ["date", "number", "population", "percent", "start", "end"]) for t in collected_texts):
            total_score += 1
            reasons.append("题型相关属性文本可见")
        else:
            reasons.append("题型相关属性线索较弱")

    if total_score >= 7:
        label = "high"
    elif total_score >= 4:
        label = "medium"
    else:
        label = "low"

    return {
        "match_label": label,
        "score": total_score,
        "reasons": reasons,
        "topic_metrics": topic_metrics,
        "answer_visible": answer_visible,
        "answer": answer,
        "compositionality_type": ctype,
    }


def build_dataset():
    print("=" * 70)
    print("高匹配样本构建")
    print("=" * 70)
    print(f"数据集: {DATASET_FILE}")
    print(f"采样数量: {SAMPLE_SIZE}")
    print(f"采样策略: {SAMPLE_STRATEGY}")

    with open(DATASET_PATH, "r", encoding="utf-8", errors="ignore") as f:
        all_data = json.load(f)

    valid = [d for d in all_data if d.get("topic_entity")]
    if SAMPLE_STRATEGY == "balanced_by_compositionality":
        samples = select_balanced_samples(valid, SAMPLE_SIZE)
    else:
        samples = valid[:SAMPLE_SIZE] if SAMPLE_SIZE > 0 else valid

    graph = _get_rdflib_graph()
    if graph is None:
        raise RuntimeError("未能加载本地知识图谱，无法构建高匹配样本")

    labeled = []
    stats = {
        "total": len(samples),
        "high": 0,
        "medium": 0,
        "low": 0,
    }
    type_stats = defaultdict(lambda: {"total": 0, "high": 0, "medium": 0, "low": 0})

    for idx, item in enumerate(samples, 1):
        result = score_sample(graph, item)
        merged = dict(item)
        merged["match_info"] = result
        labeled.append(merged)

        label = result["match_label"]
        ctype = result["compositionality_type"]
        stats[label] += 1
        type_stats[ctype]["total"] += 1
        type_stats[ctype][label] += 1

        print(f"[{idx:03d}/{len(samples)}] {item.get('ID', 'unknown')} -> {label} (score={result['score']})")

    high_samples = [x for x in labeled if x.get("match_info", {}).get("match_label") == "high"]

    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(OUTPUT_DIR, f"high_match_samples_{now_str}.json")
    md_path = os.path.join(OUTPUT_DIR, f"high_match_samples_{now_str}.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "build_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "dataset": DATASET_FILE,
                "sample_size": len(samples),
                "sample_strategy": SAMPLE_STRATEGY,
                "stats": stats,
                "type_stats": dict(type_stats),
                "high_match_count": len(high_samples),
                "items": labeled,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 高匹配样本构建结果\n\n")
        f.write(f"- 构建时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- 数据集: {DATASET_FILE}\n")
        f.write(f"- 实际样本数: {len(samples)}\n")
        f.write(f"- 采样策略: {SAMPLE_STRATEGY}\n\n")

        f.write("## 总体分布\n\n")
        f.write(f"- High: {stats['high']}\n")
        f.write(f"- Medium: {stats['medium']}\n")
        f.write(f"- Low: {stats['low']}\n")
        f.write(f"- 导出高匹配样本数: {len(high_samples)}\n\n")

        f.write("## 按题型分布\n\n")
        f.write("| 题型 | 总数 | High | Medium | Low |\n")
        f.write("|------|------|------|--------|-----|\n")
        for ctype in sorted(type_stats):
            row = type_stats[ctype]
            f.write(f"| {ctype} | {row['total']} | {row['high']} | {row['medium']} | {row['low']} |\n")

        f.write("\n## High 样本示例\n\n")
        for item in high_samples[:10]:
            info = item.get("match_info", {})
            f.write(f"- `{item.get('ID', '')}` | `{item.get('compositionality_type', '')}` | 分数={info.get('score', 0)} | 问题：{item.get('question', '')}\n")

    print("\n构建完成")
    print(f"JSON: {json_path}")
    print(f"MD  : {md_path}")
    return json_path, md_path


if __name__ == "__main__":
    build_dataset()


