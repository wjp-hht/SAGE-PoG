# NOTE: service address and credentials are read from environment
#       variables (LLM_BASE_URL / LLM_API_KEY / LLM_MODEL).
#       This file was sanitised before public release.
# -*- coding: utf-8 -*-
"""
KG 查询层 — 对接 rdflib 本地 NT 文件或远程 SPARQL
为 pog_core.py 提供统一接口：
  - get_1hop_neighbors(entity_id) -> list[dict]
  - get_name(entity_id) -> str
  - _get_rdflib_graph() / _reset_rdflib_graph()
"""

import os
import re
import sys
import logging
import threading
import warnings
warnings.filterwarnings("ignore", message=".*Failed to convert Literal.*")
warnings.filterwarnings("ignore", message=".*ISO8601Error.*")
logging.getLogger("rdflib").setLevel(logging.CRITICAL)
from rdflib import ConjunctiveGraph, URIRef, Literal

NS = "http://rdf.freebase.com/ns/"
_PREFIX_RE = re.compile(r"^(ns:|http://rdf\.freebase\.com/ns/)")

_g = None
_lock = threading.Lock()
_name_cache = {}


# ================================================================
# 图加载 / 重置
# ================================================================
def _resolve_nt_path():
    """按优先级查找 NT 文件：环境变量 > config > 平台默认路径"""
    candidates = [
        os.getenv("NT_FILE", "").strip(),
    ]
    try:
        from config import NT_FILE
        candidates.append(NT_FILE)
    except Exception:
        pass
    candidates += [
        "/opt/notebook/workspace/cc/military_kg_with_names.nt",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "cwq_topic_subgraph_targetA.nt"),
    ]
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return ""


def _get_rdflib_graph():
    global _g
    if _g is not None:
        return _g
    with _lock:
        if _g is not None:
            return _g
        path = _resolve_nt_path()
        if not path:
            print("[KG] 未找到 NT 文件，图谱为空")
            return None
        print(f"\n[KG] 正在加载图谱: {path}")
        print("[KG] (抑制rdflib日期解析警告...)")
        tmp = ConjunctiveGraph()
        # 临时屏蔽stderr，rdflib解析日期会直接print traceback
        _orig_stderr = sys.stderr
        sys.stderr = open(os.devnull, "w")
        try:
            tmp.parse(path, format="nt")
        finally:
            sys.stderr.close()
            sys.stderr = _orig_stderr
        print(f"[KG] 加载成功: {len(tmp)} 条三元组")
        _g = tmp
        return _g


def _reset_rdflib_graph():
    global _g, _name_cache
    with _lock:
        _g = None
        _name_cache = {}


# ================================================================
# ID 规范化
# ================================================================
def _normalize_id(raw):
    """将任意格式实体ID统一为 m/xxxxx 形式（无前缀）"""
    s = str(raw).strip()
    s = _PREFIX_RE.sub("", s)
    if s.startswith("m.") and "/" not in s:
        s = "m/" + s[2:]
    return s


def _to_uri(entity_id):
    """将实体ID转为完整URI。NT文件中用点号(m.xxx)，所以需要确保点号格式。"""
    eid = _normalize_id(entity_id)
    # NT文件中实体URI用点号: <http://rdf.freebase.com/ns/m.034rd>
    # 如果normalize后变成了 m/xxx，需要转回 m.xxx
    if eid.startswith("m/"):
        eid = "m." + eid[2:]
    return URIRef(NS + eid)


def _from_uri(uri_str):
    """URI -> 短ID（去掉ns前缀）"""
    s = str(uri_str)
    if s.startswith(NS):
        return s[len(NS):]
    return s


def _short_rel(rel_str):
    """关系URI -> 短名"""
    s = str(rel_str)
    if s.startswith(NS):
        return s[len(NS):]
    return s


def _is_entity_uri(node):
    """判断是否为实体URI（排除字面量和非freebase URI）"""
    if isinstance(node, Literal):
        return False
    s = str(node)
    return s.startswith(NS) and not s.endswith("/ns/")


# ================================================================
# 核心查询接口
# ================================================================
def get_1hop_neighbors(entity_id):
    """
    获取实体的所有1跳邻居（出边+入边），返回结构化列表。
    对齐 PoG 论文 Appendix D.1 SPARQL 查询。

    Returns:
        list[dict]: 每个元素 {
            'relation': str,           # 关系短名
            'connectedEntity': str,    # 邻居实体ID
            'connectedEntityName': str,# 邻居实体名称
            'direction': str,          # 'tail'(出边) 或 'head'(入边)
        }
    """
    g = _get_rdflib_graph()
    if g is None:
        return []

    subj = _to_uri(entity_id)
    results = []

    for p, o in g.predicate_objects(subj):
        if not _is_entity_uri(o):
            continue
        rel = _short_rel(p)
        eid = _from_uri(o)
        results.append({
            "relation": rel,
            "connectedEntity": eid,
            "connectedEntityName": get_name(eid),
            "direction": "tail",
        })

    for s_in, p in g.subject_predicates(subj):
        if not _is_entity_uri(s_in):
            continue
        rel = _short_rel(p)
        eid = _from_uri(s_in)
        results.append({
            "relation": rel,
            "connectedEntity": eid,
            "connectedEntityName": get_name(eid),
            "direction": "head",
        })

    return results


def get_relations(entity_id):
    """获取实体的所有关系（去重），返回 list[str]"""
    neighbors = get_1hop_neighbors(entity_id)
    rels = set()
    for n in neighbors:
        r = n["relation"]
        if n["direction"] == "head":
            r = "^" + r
        rels.add(r)
    return sorted(rels)


def get_name(entity_id):
    """获取实体的可读名称，带缓存"""
    eid = _normalize_id(entity_id)
    if eid in _name_cache:
        return _name_cache[eid]

    g = _get_rdflib_graph()
    if g is None:
        _name_cache[eid] = eid
        return eid

    # 确保URI用点号格式
    eid_dot = ("m." + eid[2:]) if eid.startswith("m/") else eid
    subj = URIRef(NS + eid_dot)
    name_preds = [
        URIRef(NS + "type.object.name"),
        URIRef("http://www.w3.org/2000/01/rdf-schema#label"),
    ]
    for pred in name_preds:
        for o in g.objects(subj, pred):
            val = str(o).strip()
            if val and not val.startswith("http"):
                lang = getattr(o, "language", None)
                if lang is None or lang == "en":
                    _name_cache[eid] = val
                    return val

    owl_same = URIRef("http://www.w3.org/2002/07/owl#sameAs")
    for o in g.objects(subj, owl_same):
        val = str(o).strip()
        if val and not val.startswith("http://rdf.freebase"):
            _name_cache[eid] = val
            return val

    _name_cache[eid] = eid
    return eid


# ================================================================
# 兼容旧接口（供 run_experiment.py 等引用）
# ================================================================
def get_neighbors(eid):
    return [(n["relation"] if n["direction"] == "tail" else "^" + n["relation"],
             n["connectedEntity"])
            for n in get_1hop_neighbors(eid)]

def get_neighbor_nodes_and_relations(eid):
    return get_neighbors(eid)

def id2entity_name_or_type(entity_id):
    return get_name(entity_id)

def execute_sparql(q):
    return []

def replace_relation_prefix(rels):
    return rels

def replace_entities_prefix(ents):
    return ents
