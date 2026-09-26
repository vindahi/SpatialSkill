#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""技能进化反思器客户端（对齐 EmbodiSkill 的 skill_model 角色）。

反思 / 技能修订 / 重构全部由该 LLM 完成。执行器（Qwen VLM）只负责答题。
两种后端（C.SKILL_LLM_BACKEND）：
- "vllm"     本地常驻 vLLM（OpenAI 兼容），默认，无需 API key（Qwen3-8B 反思器）。
- "deepseek" DeepSeek 云 API（需 DEEPSEEK_API_KEY）。
接口：
- chat(system, user): 单轮对话，返回文本。
- chat_json(system, user): 期望返回 JSON，内置去 code-fence + 容错解析 + 重试。
- parallel_map(items, fn): 窗口内反思并发（库冻结时安全）。
"""
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from . import config as C


_local_endpoint_lock = threading.Lock()
_local_endpoint_index = 0


class DeepSeekError(RuntimeError):
    pass


def _is_vllm():
    return C.SKILL_LLM_BACKEND == "vllm"


def _api_key():
    """云后端的 API key（按 backend 读对应环境变量）。"""
    if C.SKILL_LLM_BACKEND == "siliconflow":
        key = os.environ.get("SILICONFLOW_API_KEY", "").strip()
        env_name = "SILICONFLOW_API_KEY"
    else:
        key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        env_name = "DEEPSEEK_API_KEY"
    if not key:
        raise DeepSeekError(
            f"环境变量 {env_name} 未设置。技能进化的反思/修订依赖该云后端，"
            f"请先 `export {env_name}=...` 再运行（或 source src/env.sh）。")
    return key


def _endpoint():
    """返回当前后端的 (url, model, headers, extra_payload)。"""
    if _is_vllm():
        # 本地 vLLM OpenAI 兼容：无需鉴权；Qwen3.5 需显式关思维 + 强制 JSON 输出。
        global _local_endpoint_index
        urls = C.SKILL_LLM_URLS or (C.SKILL_LLM_URL,)
        with _local_endpoint_lock:
            url = urls[_local_endpoint_index % len(urls)]
            _local_endpoint_index += 1
        return (url, C.SKILL_LLM_MODEL,
                {"Content-Type": "application/json"},
                {"chat_template_kwargs": {"enable_thinking": False},
                 "response_format": {"type": "json_object"}})
    if C.SKILL_LLM_BACKEND == "siliconflow":
        # SiliconFlow 将 Qwen 的思维链开关作为请求顶层字段，而非
        # vLLM 的 chat_template_kwargs。关闭它可避免内部推理耗尽 JSON
        # 输出预算，导致 finish_reason=length 且 content 为空。
        return (C.SF_URL, C.SF_MODEL,
                {"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"},
                {"enable_thinking": C.SF_ENABLE_THINKING,
                 "response_format": {"type": "json_object"}})
    return (C.DS_URL, C.DS_MODEL,
            {"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"},
            {"response_format": {"type": "json_object"}})


def chat(system, user, temperature=None, max_tokens=1024):
    """单轮 chat completion，返回 assistant 文本。带指数退避重试。
    若 finish_reason=length（max_tokens 截断），视为失败触发重试。"""
    import requests
    url, model, headers, extra = _endpoint()
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": C.DS_TEMPERATURE if temperature is None else temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    payload.update(extra)
    last = None
    for attempt in range(1, C.DS_RETRIES + 1):
        try:
            r = requests.post(url, headers=headers, json=payload,
                              timeout=C.DS_TIMEOUT)
            if r.status_code == 200:
                data = r.json()
                choice = data["choices"][0]
                finish = choice.get("finish_reason", "")
                content = choice["message"]["content"]
                if finish == "length":
                    # max_tokens 截断——JSON 不完整，重试（下次采样可能更简洁）
                    last = f"输出被截断(finish_reason=length, len={len(content)})，token预算={max_tokens}"
                    time.sleep(min(2 ** attempt, 20))
                    continue
                return content
            last = f"HTTP {r.status_code}: {r.text[:300]}"
        except Exception as e:  # 网络抖动 / 超时
            last = repr(e)
        time.sleep(min(2 ** attempt, 20))
    raise DeepSeekError(f"反思器请求失败（{C.DS_RETRIES} 次重试后，后端={C.SKILL_LLM_BACKEND}）：{last}")


def _strip_code_fences(text):
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def parse_json(text):
    """从 LLM 文本里稳健抽取 JSON 对象。strict=False 容忍字符串内裸控制字符。"""
    cleaned = _strip_code_fences(text)
    try:
        return json.loads(cleaned, strict=False)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if m:
            return json.loads(m.group(0), strict=False)
        raise


def chat_json(system, user, max_tokens=1024, attempts=3):
    """要求 LLM 返回 JSON；解析失败重试。返回 (payload_or_None, err)。"""
    last = None
    for i in range(1, attempts + 1):
        try:
            resp = chat(system, user, max_tokens=max_tokens)
            return parse_json(resp), None
        except Exception as e:
            last = e
            print(f"[llm] JSON 解析/请求失败 {i}/{attempts}: {e}", flush=True)
    return None, last


def parallel_map(items, fn, workers=None):
    """窗口内反思并发；items 少或 workers<=1 时退化为串行。"""
    items = list(items)
    w = workers or C.LLM_CONCURRENCY
    if len(items) <= 1 or w <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=min(w, len(items))) as ex:
        return list(ex.map(fn, items))


def available():
    """探测反思器是否就绪。

    - vllm 后端：每个本地副本都须通过 GET {base}/models 健康检查。
    - deepseek 后端：检查 DEEPSEEK_API_KEY 是否存在（不发请求）。
    """
    if _is_vllm():
        import requests
        try:
            for url in C.SKILL_LLM_URLS or (C.SKILL_LLM_URL,):
                base = url.rsplit("/chat/completions", 1)[0].rstrip("/")
                response = requests.get(base + "/models", timeout=5)
                if response.status_code != 200:
                    return False
                ids = [item.get("id") for item in response.json().get("data", [])]
                if C.SKILL_LLM_MODEL not in ids:
                    return False
            return True
        except Exception:
            return False
    # 云 API（siliconflow / deepseek）：检查对应 key 是否存在（不发请求）。
    if C.SKILL_LLM_BACKEND == "siliconflow":
        return bool(os.environ.get("SILICONFLOW_API_KEY", "").strip())
    return bool(os.environ.get("DEEPSEEK_API_KEY", "").strip())
