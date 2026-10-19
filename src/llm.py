# NOTE: service address and credentials are read from environment
#       variables (LLM_BASE_URL / LLM_API_KEY / LLM_MODEL).
#       This file was sanitised before public release.
# -*- coding: utf-8 -*-
"""
LLM统一调用模块（DSH 修订版）
优先使用环境变量配置的OpenAI兼容API，兜底保留原始逻辑

环境变量配置：
  LLM_BASE_URL  - API基础URL，例如 os.environ.get('LLM_BASE_URL', 'http://localhost:11434/v1')
  LLM_API_KEY   - API密钥
  LLM_MODEL     - 模型名称，例如 qwen3:32b

可选调参（均有默认值，不设也能跑）：
  LLM_TIMEOUT / LLM_TIMEOUT_RESPONSES / LLM_TIMEOUT_FALLBACK   超时秒数
  LLM_MAX_RETRIES        每个接口的最大尝试次数（默认 3）
  LLM_BACKOFF_BASE       指数退避底数（默认 1.5）
  LLM_BACKOFF_CAP        退避上限秒数（默认 20）
  LLM_MAX_TOKENS / LLM_MAX_TOKENS_RESPONSES / LLM_MAX_TOKENS_FALLBACK
  LLM_POOL_SIZE          连接池大小（默认 32）

—— 本版相对原版的修改（均为缺陷修复）——
  [FIX-1] <think> 清理逻辑重写为 _strip_think()：
          原逻辑在 <think> 未闭合时失效（split 无分隔符返回原文，
          `if not result` 恒为假，清理代码永不执行），导致思维链泄漏。
          新逻辑分离处理「成对标签 / 未闭合开标签 / 孤儿闭标签」三类情形。
  [FIX-2] 兜底分支补上同样的清理（原版兜底分支完全没有清理）。
  [FIX-3] 用模块级 requests.Session + HTTPAdapter 连接池，替代每次 requests.post
          新建连接；进程退出时自动复用。
  [FIX-4] 重试策略化：网络异常与 408/409/425/429/500/502/503/504 才重试，
          400/401/403/404 等不可重试错误立即返回；退避改为指数 + 抖动。
  [FIX-5] 统计拆分为 logic_calls（逻辑调用）/ http_requests（真实HTTP请求）
          / retries / errors / 分状态码计数 / 延迟累计，供实验统计使用。
  [FIX-6] 超时、重试次数、max_tokens、连接池大小改为可配置。
  [FIX-7] 返回值与日志行格式保持向后兼容：
          仍打印 "[LLM] <model> 响应完成"，run_LLM 签名不变。
"""

import os
import random
import re
import threading
import time

import requests
from requests.adapters import HTTPAdapter

try:
    from urllib3.util.retry import Retry  # noqa: F401  (保留依赖探测)
except Exception:  # pragma: no cover
    Retry = None


# ============================================================
# 配置
# ============================================================
def _env_float(name, default):
    try:
        v = os.getenv(name, "").strip()
        return float(v) if v else float(default)
    except Exception:
        return float(default)


def _env_int(name, default):
    try:
        v = os.getenv(name, "").strip()
        return int(v) if v else int(default)
    except Exception:
        return int(default)


TIMEOUT_MAIN = _env_float("LLM_TIMEOUT", 180)
TIMEOUT_RESPONSES = _env_float("LLM_TIMEOUT_RESPONSES", 120)
TIMEOUT_FALLBACK = _env_float("LLM_TIMEOUT_FALLBACK", 120)

MAX_RETRIES = max(1, _env_int("LLM_MAX_RETRIES", 3))
BACKOFF_BASE = _env_float("LLM_BACKOFF_BASE", 1.5)
BACKOFF_CAP = _env_float("LLM_BACKOFF_CAP", 20.0)

MAX_TOKENS_MAIN = _env_int("LLM_MAX_TOKENS", 2048)
MAX_TOKENS_RESPONSES = _env_int("LLM_MAX_TOKENS_RESPONSES", 1024)
MAX_TOKENS_FALLBACK = _env_int("LLM_MAX_TOKENS_FALLBACK", 512)

POOL_SIZE = max(1, _env_int("LLM_POOL_SIZE", 32))

# [FIX-13] 采样随机种子：设为非空整数后随每次请求发送，用于重复性实验。
# 实测目标服务（qwen3:32b）支持 seed：同 seed 多次输出完全一致，不同 seed 输出不同，
# 不传 seed 时每次不同。留空 = 保持原随机采样行为。
_SEED_ENV = "LLM_SEED"

# 可重试的 HTTP 状态码
RETRY_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


# ============================================================
# 统计（[FIX-5]）
# ============================================================
_LOCK = threading.Lock()
_STATS = {
    "logic_calls": 0,      # run_LLM 被调用的次数（= 原 _LLM_CALL_COUNTER 语义）
    "http_requests": 0,    # 真实发出的 HTTP 请求数（含重试）
    "retries": 0,          # 重试次数
    "errors": 0,           # 最终失败的逻辑调用数
    "http_4xx": 0,
    "http_5xx": 0,
    "network_errors": 0,
    "empty_results": 0,    # 返回空串的逻辑调用数
    "think_stripped": 0,   # 触发了思维链清理的响应数
    "latency_sum": 0.0,    # 成功调用的总耗时（秒）
    "latency_max": 0.0,
    "latency_n": 0,
    # [FIX-12] token 统计（补充实验第三阶段要求记录 prompt/completion/total tokens）
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "usage_reported": 0,   # 成功取到 usage 的响应数
    "usage_missing": 0,    # 响应体里没有 usage 的次数（服务不支持统计时）
}



def _bump(field, delta=1):
    with _LOCK:
        _STATS[field] = _STATS.get(field, 0) + delta


def current_seed():
    """[FIX-13] 当前生效的采样种子（None = 未设置）。每次调用都读环境变量，
    以便同一个进程内按配置切换。"""
    v = os.getenv(_SEED_ENV, "").strip()
    if not v:
        return None
    try:
        return int(v)
    except ValueError:
        return None


def _apply_seed(payload):
    """[FIX-13] 若设置了 LLM_SEED，则把 seed 写进请求体（原地修改并返回）"""
    s = current_seed()
    if s is not None:
        payload["seed"] = s
    return payload


def reset_llm_call_counter():
    """重置全部统计（保持原函数名向后兼容）"""
    with _LOCK:
        for k in list(_STATS.keys()):
            _STATS[k] = 0 if not isinstance(_STATS[k], float) else 0.0


def get_llm_call_counter():
    """逻辑调用次数（与原版语义一致）"""
    with _LOCK:
        return _STATS["logic_calls"]


def get_llm_stats():
    """完整统计快照"""
    with _LOCK:
        s = dict(_STATS)
    n = s.get("latency_n", 0)
    s["latency_avg"] = (s["latency_sum"] / n) if n else 0.0
    lc = s.get("logic_calls", 0)
    s["http_per_call"] = (s["http_requests"] / lc) if lc else 0.0
    s["tokens_per_call"] = (s["total_tokens"] / lc) if lc else 0.0
    s["seed"] = current_seed()          # [FIX-13] 便于结果文件留痕
    return s


def get_llm_token_snapshot():
    """[FIX-12] 返回累计计数快照，供调用方做「逐题增量」统计。

    返回 dict，含 prompt/completion/total tokens 与 http_requests/retries/logic_calls。
    用法：
        a = get_llm_token_snapshot()
        ... 跑一道题 ...
        b = get_llm_token_snapshot()
        本题 token = {k: b[k] - a[k] for k in b}
    """
    with _LOCK:
        return {
            "prompt_tokens": _STATS["prompt_tokens"],
            "completion_tokens": _STATS["completion_tokens"],
            "total_tokens": _STATS["total_tokens"],
            "http_requests": _STATS["http_requests"],
            "retries": _STATS["retries"],
            "logic_calls": _STATS["logic_calls"],
        }


def _record_usage(usage, kind="chat"):
    """[FIX-12] 记录一次响应的 token 用量。

    chat/兜底端点为 prompt_tokens/completion_tokens/total_tokens；
    /responses 端点为 input_tokens/output_tokens/total_tokens。
    服务未返回 usage 时计入 usage_missing，不抛异常。
    """
    if not isinstance(usage, dict):
        _bump("usage_missing")
        return
    if kind == "responses":
        p = usage.get("input_tokens")
        c = usage.get("output_tokens")
    else:
        p = usage.get("prompt_tokens")
        c = usage.get("completion_tokens")
    t = usage.get("total_tokens")
    if p is None and c is None and t is None:
        _bump("usage_missing")
        return
    try:
        p = int(p or 0)
        c = int(c or 0)
        t = int(t) if t is not None else p + c
    except Exception:
        _bump("usage_missing")
        return
    with _LOCK:
        _STATS["prompt_tokens"] += p
        _STATS["completion_tokens"] += c
        _STATS["total_tokens"] += t
        _STATS["usage_reported"] += 1


def format_llm_stats(prefix="[LLM统计]"):
    s = get_llm_stats()
    return (f"{prefix} 逻辑调用={s['logic_calls']} 真实HTTP={s['http_requests']} "
            f"(每次{s['http_per_call']:.2f}) 重试={s['retries']} 失败={s['errors']} "
            f"4xx={s['http_4xx']} 5xx={s['http_5xx']} 网络异常={s['network_errors']} "
            f"空返回={s['empty_results']} 清理思维链={s['think_stripped']} | "
            f"tokens: prompt={s['prompt_tokens']} completion={s['completion_tokens']} "
            f"total={s['total_tokens']} (每次{s['tokens_per_call']:.0f}) "
            f"[有usage {s['usage_reported']} / 无usage {s['usage_missing']}] | "
            f"延迟均值={s['latency_avg']:.2f}s 最大={s['latency_max']:.2f}s")


# ============================================================
# 会话与连接池（[FIX-3]）
# ============================================================
_SESSION = None
_SESSION_LOCK = threading.Lock()


def _get_session():
    global _SESSION
    if _SESSION is None:
        with _SESSION_LOCK:
            if _SESSION is None:
                s = requests.Session()
                adapter = HTTPAdapter(
                    pool_connections=POOL_SIZE,
                    pool_maxsize=POOL_SIZE,
                    max_retries=0,        # 重试由本模块控制，避免双重重试
                )
                s.mount("http://", adapter)
                s.mount("https://", adapter)
                _SESSION = s
    return _SESSION


def _post(url, headers, payload, timeout):
    """统一的 POST：计数 + 计时，使用共享连接池"""
    _bump("http_requests")
    t0 = time.time()
    try:
        resp = _get_session().post(url, headers=headers, json=payload, timeout=timeout)
        return resp, time.time() - t0
    except Exception:
        with _LOCK:
            _STATS["network_errors"] += 1
        raise


def _record_http_status(code):
    if 400 <= code < 500:
        _bump("http_4xx")
    elif 500 <= code < 600:
        _bump("http_5xx")


def _sleep_backoff(attempt):
    """指数退避 + 抖动（[FIX-4]）"""
    d = min(BACKOFF_CAP, BACKOFF_BASE ** attempt)
    time.sleep(d * (0.5 + random.random()))


# ============================================================
# 思维链清理（[FIX-1]）
# ============================================================
_THINK_PAIRED = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think\b[^>]*>", re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think\s*>", re.IGNORECASE)


def strip_think(text):
    """
    稳健清理思维链标签，返回 (清理后文本, 是否发生清理)。

    三类情形分别处理：
      1) 成对 <think>…</think>          → 整体删除
      2) 仍有未闭合 <think>…            → 从该标签起全部丢弃（多为输出被截断，其后是残缺推理，不是答案）
      3) 孤儿 </think>                  → 仅删除标签本身
    另兼容大小写与标签内空白。
    """
    if not text:
        return "", False
    original = text
    t = _THINK_PAIRED.sub("", text)
    if _THINK_OPEN.search(t):
        t = _THINK_OPEN.split(t)[0]
    t = _THINK_CLOSE.sub("", t).strip()
    changed = (t != original.strip())
    return t, changed


def _finalize(text, model_name, tag):
    """统一收尾：清理思维链 + 统计 + 日志（保持原日志行格式）"""
    t0 = time.time()
    clean, stripped = strip_think(text or "")
    if stripped:
        _bump("think_stripped")
    if not clean:
        _bump("empty_results")
    dt = time.time() - t0
    _bump("latency_sum", dt)
    _bump("latency_n")
    with _LOCK:
        if dt > _STATS["latency_max"]:
            _STATS["latency_max"] = dt
    print(f"[LLM] {model_name} {tag}响应完成")
    return clean


# ============================================================
# 主入口
# ============================================================
def run_LLM(prompt, model="gpt4", temperature=0.4):
    """
    统一LLM调用入口：
    1) 优先使用环境变量指定的 OpenAI 兼容 API
    2) 兜底使用本地 localhost:8000 或默认百度千帆
    签名与返回值与原版完全兼容。
    """
    _bump("logic_calls")

    env_base_url = os.getenv("LLM_BASE_URL", "").strip()
    env_api_key = os.getenv("LLM_API_KEY", "").strip()
    env_model = os.getenv("LLM_MODEL", "").strip()

    # ===== 优先：环境变量配置的 OpenAI 兼容 API =====
    if env_base_url and env_api_key and env_model:
        chat_url = env_base_url.rstrip('/') + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {env_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": env_model,
            "messages": [
                {"role": "system",
                 "content": "You are an AI assistant that helps people find information."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": MAX_TOKENS_MAIN,
        }
        _apply_seed(payload)                     # [FIX-13]
        last_status = None
        for attempt in range(MAX_RETRIES):
            try:
                resp, _ = _post(chat_url, headers, payload, TIMEOUT_MAIN)
                if resp.status_code == 200:
                    try:
                        _data = resp.json()
                        raw = _data["choices"][0]["message"]["content"]
                        _record_usage(_data.get("usage"), "chat")   # [FIX-12]
                    except Exception as ex:
                        print(f"[LLM] 响应体解析失败: {ex}")
                        _bump("errors")
                        _bump("retries") if attempt else None
                        if attempt < MAX_RETRIES - 1:
                            _sleep_backoff(attempt)
                            continue
                        break
                    return _finalize(raw, env_model, "")
                _record_http_status(resp.status_code)
                last_status = resp.status_code
                print(f"[LLM] HTTP {resp.status_code}: {resp.text[:200]}")
                if resp.status_code not in RETRY_STATUS:
                    break                      # 不可重试，直接跳出
                if attempt < MAX_RETRIES - 1:
                    _bump("retries")
                    _sleep_backoff(attempt)
            except Exception as e:
                print(f"[LLM] 环境变量API错误(第{attempt+1}次): {e}")
                if attempt < MAX_RETRIES - 1:
                    _bump("retries")
                    _sleep_backoff(attempt)

        # [FIX-8] 鉴权类错误（401/403）下 /responses 必然同样失败，不再空跑一次
        if last_status in (401, 403):
            print(f"[LLM] chat 端点返回 {last_status}（鉴权失败），跳过 /responses 兜底")
            _bump("errors")
            _bump("empty_results")
            return ""

        # chat/completions 失败后，兜底到 responses 接口
        responses_url = env_base_url.rstrip('/') + "/responses"
        payload_r = {
            "model": env_model,
            "input": [{"role": "user",
                       "content": [{"type": "input_text", "text": prompt}]}],
            "max_output_tokens": MAX_TOKENS_RESPONSES,
        }
        _apply_seed(payload_r)                   # [FIX-13]
        try:
            resp, _ = _post(responses_url, headers, payload_r, TIMEOUT_RESPONSES)
            if resp.status_code == 200:
                data = resp.json()
                _record_usage(data.get("usage"), "responses")       # [FIX-12]
                texts = []
                for item in data.get("output", []):
                    if item.get("type") == "message":
                        for c in item.get("content", []):
                            if c.get("type") in ("output_text", "text"):
                                t = c.get("text", "")
                                if t:
                                    texts.append(t)
                if not texts:
                    for item in data.get("output", []):
                        if item.get("type") == "reasoning":
                            for s in item.get("summary", []):
                                if s.get("type") == "summary_text":
                                    t = s.get("text", "")
                                    if t:
                                        texts.append(t)
                return _finalize("\n".join(texts), env_model, "responses接口")
            _record_http_status(resp.status_code)
            print(f"[LLM] responses接口错误: HTTP {resp.status_code}")
        except Exception as e:
            print(f"[LLM] responses接口异常: {e}")
        _bump("errors")
        _bump("empty_results")
        return ""

    # ===== 兜底：本地 localhost 或 百度千帆 =====
    fallback_base = os.getenv("FALLBACK_BASE_URL", "http://localhost:8000/v1").rstrip('/')
    fallback_key = os.getenv("FALLBACK_API_KEY", "EMPTY")
    actual_model = (os.getenv("FALLBACK_MODEL", "deepseek-v3.2")
                    if "gpt" in model.lower() else model)

    headers = {
        "Authorization": f"Bearer {fallback_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": actual_model,
        "messages": [
            {"role": "system",
             "content": "You are an AI assistant that helps people find information."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": MAX_TOKENS_FALLBACK,
    }
    _apply_seed(payload)                         # [FIX-13]

    chat_url = fallback_base + "/chat/completions"
    for attempt in range(MAX_RETRIES):
        try:
            resp, _ = _post(chat_url, headers, payload, TIMEOUT_FALLBACK)
            if resp.status_code == 200:
                try:
                    _data2 = resp.json()
                    raw = _data2["choices"][0]["message"]["content"]
                    _record_usage(_data2.get("usage"), "chat")      # [FIX-12]
                except Exception as ex:
                    print(f"[LLM] 兜底响应体解析失败: {ex}")
                    if attempt < MAX_RETRIES - 1:
                        _bump("retries")
                        _sleep_backoff(attempt)
                        continue
                    break
                # [FIX-2] 兜底分支同样清理思维链
                return _finalize(raw, actual_model, "兜底")
            _record_http_status(resp.status_code)
            print(f"[LLM] 兜底HTTP {resp.status_code}: {resp.text[:180]}")
            if resp.status_code not in RETRY_STATUS:
                break
            if attempt < MAX_RETRIES - 1:
                _bump("retries")
                _sleep_backoff(attempt)
        except Exception as e:
            print(f"[LLM] 兜底API错误(第{attempt+1}次): {e}")
            if attempt < MAX_RETRIES - 1:
                _bump("retries")
                _sleep_backoff(attempt)
    _bump("errors")
    _bump("empty_results")
    return ""
