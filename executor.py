#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VLM 执行器调度（默认 Qwen3-VL-8B-Instruct，EXEC_MODEL_NAME 可切 9B 等）。

三种后端：
  1) 常驻 vLLM 服务（优先，开源模型）：run_executor(jobs) 自动读 endpoints.json
  2) 云 API（闭源模型 GPT/Claude/Gemini）：EXEC_BACKEND=api，见 run_executor_api
  3) 子进程多卡回退：每卡一个 vllm 子进程处理一个分片（--worker 内部入口）
"""
import argparse
from contextlib import contextmanager
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import config as C
from . import data as D


# --------------------------- 常驻服务客户端 ---------------------------
def service_endpoints():
    """读服务端点；未启动返回 None。"""
    if not os.path.exists(C.SERVICE_FILE):
        return None
    try:
        info = json.load(open(C.SERVICE_FILE))
        eps = info.get("endpoints") or []
        return eps or None
    except Exception:
        return None


def _health_ok(base):
    import requests
    try:
        r = requests.get(base.rstrip("/") + "/models", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def _serves_model(base, model_name):
    """确认健康端点实际提供当前执行器模型，避免复用旧服务时静默串模型。"""
    import requests
    try:
        r = requests.get(base.rstrip("/") + "/models", timeout=5)
        if r.status_code != 200:
            return False
        return any(item.get("id") == model_name for item in r.json().get("data", []))
    except Exception:
        return False


def _require_matching_service(endpoints):
    matching = [base for base in endpoints if _serves_model(base, C.SERVED_MODEL_NAME)]
    if matching:
        return matching
    raise RuntimeError(
        f"常驻服务未提供当前执行器模型 {C.SERVED_MODEL_NAME!r}；"
        "请先停止旧服务并运行 bash src/serve_vllm.sh")


def _wait_for_matching_service(timeout=None):
    """等待由副本托管器重启的服务恢复健康。"""
    timeout = C.EXECUTOR_SERVICE_WAIT_SECONDS if timeout is None else timeout
    deadline = time.monotonic() + max(0, timeout)
    announced = False
    while True:
        eps = service_endpoints()
        if eps:
            healthy = [e for e in eps if _health_ok(e)]
            matching = [e for e in healthy if _serves_model(e, C.SERVED_MODEL_NAME)]
            if matching:
                return matching
            if not announced:
                print(f"[service] waiting up to {timeout}s for {C.SERVED_MODEL_NAME!r} replicas...",
                      flush=True)
                announced = True
        if time.monotonic() >= deadline:
            return []
        time.sleep(5)


def _one_request(base, job, enable_thinking, retries=None):
    import requests
    payload = {
        "model": C.SERVED_MODEL_NAME,
        "messages": D.job_to_messages(job),
        "temperature": 0.0,
        "max_tokens": C.GEN_MAX_TOKENS,
        "frequency_penalty": C.GEN_FREQ_PENALTY,
        "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
    }
    url = base.rstrip("/") + "/chat/completions"
    last = ""
    # A supervisor restart includes model loading.  Keep a request alive through
    # that short outage rather than silently recording it as an empty answer.
    retries = retries or max(3, C.EXECUTOR_SERVICE_WAIT_SECONDS // 5)
    for _ in range(retries):
        try:
            r = requests.post(url, json=payload, timeout=600)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            if 400 <= r.status_code < 500:
                break
        except Exception as e:
            last = repr(e)
        time.sleep(5)
    return None if last else ""


_ANS_RE = re.compile(r"ANSWER\s*[:\uff1a]\s*[A-F]", re.IGNORECASE)


def _ensure_answer(base, job, raw, enable_thinking):
    """二段式兜底：首次输出无 ANSWER 行（多为截断/死循环）时，
    把已有推理作为 assistant 上下文追加一次短请求，强制只出答案行。"""
    if not raw or _ANS_RE.search(raw):
        return raw
    import requests
    msgs = D.job_to_messages(job)
    msgs.append({"role": "assistant", "content": raw})
    msgs.append({"role": "user", "content":
                 "Your reasoning above was cut off. Do NOT continue reasoning. "
                 "Based on it, respond with exactly one line:\nANSWER: <letter>"})
    payload = {
        "model": C.SERVED_MODEL_NAME,
        "messages": msgs,
        "temperature": 0.0,
        "max_tokens": C.ANSWER_RETRY_MAX_TOKENS,
        "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
    }
    try:
        r = requests.post(base.rstrip("/") + "/chat/completions",
                          json=payload, timeout=120)
        if r.status_code == 200:
            tail = r.json()["choices"][0]["message"]["content"] or ""
            if tail.strip():
                return raw + "\n" + tail.strip()
    except Exception:
        pass
    return raw


def run_executor_service(jobs, endpoints, enable_thinking=False,
                         concurrency=None, progress_every=200):
    """并发轮询多个常驻副本推理，返回 {id: {answer, op_seq, raw}}。"""
    n_ep = len(endpoints)
    workers = (concurrency or C.SERVICE_CONCURRENCY) * n_ep
    results = {}
    done = [0]

    def task(i_job):
        i, job = i_job
        base = endpoints[i % n_ep]
        raw = _one_request(base, job, enable_thinking)
        if raw is None:
            return job["id"], {"id": job["id"], "answer": None, "op_seq": [], "raw": ""}
        raw = _ensure_answer(base, job, raw, enable_thinking)
        ans = D.parse_answer(raw, job["valid_letters"])
        return job["id"], {"id": job["id"], "answer": ans,
                            "op_seq": D.parse_op_sequence(raw), "raw": raw}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for rid, res in ex.map(task, enumerate(jobs)):
            results[rid] = res
            done[0] += 1
            if progress_every and done[0] % progress_every == 0:
                print(f"  [service] {done[0]}/{len(jobs)} done", flush=True)
    return results


def infer_one(job, enable_thinking=False, endpoints=None):
    """串行：向单个常驻副本发一条请求，返回 {id, answer, op_seq, raw}。"""
    eps = endpoints or service_endpoints()
    if not eps:
        raise RuntimeError("常驻服务未启动，请先运行 src/serve_vllm.sh")
    eps = _require_matching_service(eps)
    raw = _one_request(eps[0], job, enable_thinking)
    if raw is None:
        return {"id": job["id"], "answer": None, "op_seq": [], "raw": ""}
    raw = _ensure_answer(eps[0], job, raw, enable_thinking)
    return {"id": job["id"], "answer": D.parse_answer(raw, job["valid_letters"]),
            "op_seq": D.parse_op_sequence(raw), "raw": raw}


# --------------------------- 云 API 执行器（闭源模型 GPT/Claude/Gemini 等） ---------------------------
class ExecutorAPIError(RuntimeError):
    """API 执行器在可配置的最大尝试次数后仍无法完成请求。"""


class ExecutorInterrupted(RuntimeError):
    """用户中断了正在等待重试的 API 执行器。"""


@contextmanager
def _cancel_api_retries_on_sigint(cancel_event):
    """首个 Ctrl-C 取消等待中的重试，第二个 Ctrl-C 保留强制中断语义。"""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    try:
        previous = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, lambda _signum, _frame: None)
    except ValueError:  # 非主线程 / 嵌入式调用环境不允许安装信号处理器
        yield
        return

    def handle_sigint(_signum, _frame):
        if cancel_event.is_set():
            raise KeyboardInterrupt
        cancel_event.set()
        print("\n[api] 已收到 Ctrl-C：取消等待中的重试；正在进行的 HTTP 请求结束后将退出。",
              flush=True)

    signal.signal(signal.SIGINT, handle_sigint)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


def _openai_chat_completions_url(value):
    """将 OpenAI 兼容基地址或完整地址统一为 chat completions 端点。"""
    url = str(value or "").strip().rstrip("/")
    if not url:
        return url
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return url + "/chat/completions"
    return url + "/v1/chat/completions"


def _response_summary(response):
    content_type = response.headers.get("content-type", "unknown")
    body = " ".join((response.text or "").split())[:300]
    return f"Content-Type={content_type}; body={body!r}"


def _api_build_request(job, api):
    """按一个 API 配置构建请求；不同 API 可使用不同协议和模型。"""
    style = api["style"]
    api_key = os.environ.get(api.get("api_key_env", ""), api.get("api_key", ""))
    url_value = api["url"]
    if style == "openai":
        # GPT 及一切 OpenAI 兼容网关（OpenRouter 等也可接 Claude/Gemini）
        url = _openai_chat_completions_url(url_value)
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        payload = {"model": api["model"],
                   "messages": D.job_to_messages_api(job),
                   "temperature": 0.0, "max_tokens": C.GEN_MAX_TOKENS}
    elif style == "anthropic":
        url = url_value
        headers = {"anthropic-version": "2023-06-01"}
        if api_key:
            headers["x-api-key"] = api_key
        content = []
        for p in job["images"]:
            mime, b64 = D.image_b64(p)
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": mime, "data": b64}})
        content.append({"type": "text", "text": job["text"]})
        payload = {"model": api["model"], "max_tokens": C.GEN_MAX_TOKENS,
                   "temperature": 0.0,
                   "messages": [{"role": "user", "content": content}]}
    elif style == "gemini":
        url = (url_value.rstrip("/")
               + f"/models/{api['model']}:generateContent")
        headers = {"x-goog-api-key": api_key} if api_key else {}
        parts = []
        for p in job["images"]:
            mime, b64 = D.image_b64(p)
            parts.append({"inline_data": {"mime_type": mime, "data": b64}})
        parts.append({"text": job["text"]})
        payload = {"contents": [{"role": "user", "parts": parts}],
                   "generationConfig": {"temperature": 0.0,
                                        "maxOutputTokens": C.GEN_MAX_TOKENS}}
    else:
        raise ValueError(f"未知 API style {style!r}（合法：openai|anthropic|gemini）")
    return url, headers, payload


def _api_extract_text(resp, style):
    """按风格从响应 JSON 提取文本。"""
    if style == "openai":
        return resp["choices"][0]["message"]["content"] or ""
    if style == "anthropic":
        return "".join(b.get("text", "") for b in resp.get("content", []))
    return "".join(p.get("text", "")
                   for p in resp["candidates"][0]["content"].get("parts", []))


def _api_retryable_status(status_code):
    """仅重试瞬态 HTTP 错误；鉴权、模型名和请求体错误应立即暴露。"""
    return status_code in (408, 409, 425, 429) or 500 <= status_code < 600


def _api_one_request(job, api, cancel_event):
    import requests
    url, headers, payload = _api_build_request(job, api)
    last = ""
    attempts = max(1, int(api["max_retries"]))
    for attempt in range(1, attempts + 1):
        if cancel_event.is_set():
            return None, "cancelled"
        try:
            r = requests.post(url, json=payload, headers=headers,
                              timeout=api["timeout"], allow_redirects=False)
            if 200 <= r.status_code < 300:
                try:
                    content = _api_extract_text(r.json(), api["style"])
                except (ValueError, KeyError, IndexError, TypeError) as e:
                    last = f"HTTP 200 malformed API response ({e}): {_response_summary(r)}"
                    retryable = False
                else:
                    return content, None
            else:
                last = f"HTTP {r.status_code}: {_response_summary(r)}"
                retryable = _api_retryable_status(r.status_code)
        except Exception as e:
            last = repr(e)
            retryable = True

        if not retryable or attempt == attempts:
            break
        print(f"  [api] {job['id']} request {attempt}/{attempts} failed: {last}; "
              f"retrying in {api['retry_delay_seconds']}s via {api['name']} "
              "(Ctrl-C cancels retries)",
              flush=True)
        if cancel_event.wait(api["retry_delay_seconds"]):
            return None, "cancelled"
    return None, last or "request cancelled"


def run_executor_api(jobs, concurrency=None, progress_every=50, result_sink=None):
    """Run jobs through an API pool with endpoint-level circuit breakers.

    Every configured endpoint contributes its concurrency to one shared
    worker pool. A job that fails on one endpoint is returned to the queue so
    another healthy endpoint can process it. Once an endpoint accumulates
    ``failure_threshold`` consecutive failed jobs, it is frozen for
    ``cooldown_seconds``. When the cooldown expires, exactly one probe is
    admitted: success restores the endpoint, failure starts a new cooldown.
    A positive ``max_freeze_cycles`` permanently removes an endpoint after
    that many failed cooldown cycles; when every endpoint is removed, the
    executor raises :class:`ExecutorAPIError`.
    """
    api_pools = tuple(C.EXEC_API_POOLS)
    if not api_pools:
        raise RuntimeError("EXEC_BACKEND=api 需要在 EXEC_API_CONFIG 中配置至少一个 API")
    if not jobs:
        return {}

    # A generic worker pool lets capacity move away from frozen URLs instead
    # of leaving their dedicated threads idle. State changes share one
    # condition so cooldown expiry, released capacity, and re-queued jobs wake
    # the waiting workers together.
    endpoint_states = []
    for api in api_pools:
        endpoint_states.append({
            "api": api,
            "capacity": max(1, int(concurrency or api["concurrency"])),
            "in_flight": 0,
            "consecutive_failures": 0,
            "frozen_until": 0.0,
            "probe_in_flight": False,
            "freeze_cycles": 0,
            "exhausted": False,
            "last_error": "",
        })

    results = {}
    cancel_event = threading.Event()
    from collections import deque

    pending = deque(jobs)
    state_changed = threading.Condition()
    result_lock = threading.Lock()
    total_workers = sum(state["capacity"] for state in endpoint_states)
    done = [0]

    def _take_work():
        """Reserve one queued job and an endpoint slot, or wait for either."""
        with state_changed:
            while True:
                if cancel_event.is_set() or not pending:
                    return None

                now = time.monotonic()
                next_wakeup = None
                for state in endpoint_states:
                    if state["exhausted"]:
                        continue
                    frozen_until = state["frozen_until"]
                    if frozen_until:
                        if now < frozen_until:
                            next_wakeup = (
                                frozen_until if next_wakeup is None
                                else min(next_wakeup, frozen_until)
                            )
                            continue
                        # A recovered endpoint is half-open: wait for every
                        # pre-freeze request to finish, then allow one probe.
                        if state["in_flight"] or state["probe_in_flight"]:
                            continue
                        state["in_flight"] += 1
                        state["probe_in_flight"] = True
                        print(
                            f"  [api] cooldown elapsed for {state['api']['name']}; "
                            "sending one recovery probe",
                            flush=True,
                        )
                        return pending.popleft(), state, True

                    if state["in_flight"] < state["capacity"]:
                        state["in_flight"] += 1
                        return pending.popleft(), state, False

                if all(state["exhausted"] for state in endpoint_states):
                    detail = "; ".join(
                        f"{state['api']['name']} after {state['freeze_cycles']} "
                        f"freeze cycle(s): {state['last_error']}"
                        for state in endpoint_states
                    )
                    raise ExecutorAPIError(f"API pool exhausted: {detail}")

                # Keep this wait short enough that Ctrl-C cancels even when
                # every endpoint is in a 15-minute cooldown.
                timeout = 1.0
                if next_wakeup is not None:
                    timeout = min(timeout, max(0.0, next_wakeup - now))
                state_changed.wait(timeout=timeout)

    def _record_success(state, was_probe):
        with state_changed:
            state["in_flight"] -= 1
            state["probe_in_flight"] = False
            # Successes admitted before a different request tripped the
            # breaker do not reopen it. Only the half-open probe can do that.
            if not state["frozen_until"] or was_probe:
                state["consecutive_failures"] = 0
                state["frozen_until"] = 0.0
                if was_probe:
                    print(f"  [api] recovery probe succeeded for {state['api']['name']}",
                          flush=True)
            state_changed.notify_all()

    def _record_failure(state, was_probe, detail):
        with state_changed:
            state["in_flight"] -= 1
            state["probe_in_flight"] = False
            api = state["api"]
            state["last_error"] = detail
            threshold = api["failure_threshold"]
            if was_probe:
                state["consecutive_failures"] = max(
                    threshold, state["consecutive_failures"] + 1)
                should_freeze = True
            elif state["frozen_until"]:
                # This request was already in flight when another failure
                # opened the breaker; do not extend that cooldown repeatedly.
                should_freeze = False
            else:
                state["consecutive_failures"] += 1
                should_freeze = state["consecutive_failures"] >= threshold

            if should_freeze:
                state["freeze_cycles"] += 1
                phase = "recovery probe failed" if was_probe else "circuit opened"
                max_cycles = api["max_freeze_cycles"]
                if max_cycles and state["freeze_cycles"] >= max_cycles:
                    state["exhausted"] = True
                    state["frozen_until"] = 0.0
                    print(
                        f"  [api] {api['name']} removed from pool after "
                        f"{state['freeze_cycles']} failed freeze cycle(s): {detail}",
                        flush=True,
                    )
                else:
                    state["frozen_until"] = time.monotonic() + api["cooldown_seconds"]
                    print(
                        f"  [api] {phase} for {api['name']} after "
                        f"{state['consecutive_failures']} consecutive failure(s); "
                        f"frozen for {api['cooldown_seconds']}s "
                        f"(cycle {state['freeze_cycles']}): {detail}",
                        flush=True,
                    )
            state_changed.notify_all()

    def task():
        while not cancel_event.is_set():
            assignment = _take_work()
            if assignment is None:
                return
            job, state, was_probe = assignment
            try:
                raw, error = _api_one_request(job, state["api"], cancel_event)
            except Exception as exc:
                raw, error = None, repr(exc)

            if raw is None:
                if error == "cancelled" or cancel_event.is_set():
                    with state_changed:
                        state["in_flight"] -= 1
                        state["probe_in_flight"] = False
                        state_changed.notify_all()
                    return
                _record_failure(state, was_probe, error)
                # The task remains pending and can move to a healthy URL.
                with state_changed:
                    pending.appendleft(job)
                    state_changed.notify_all()
                continue

            _record_success(state, was_probe)
            result = {
                "id": job["id"],
                "answer": D.parse_answer(raw, job["valid_letters"]),
                "op_seq": D.parse_op_sequence(raw),
                "raw": raw,
                "api": state["api"]["name"],
            }
            with result_lock:
                results[job["id"]] = result
                if result_sink is not None:
                    result_sink(result)
                done[0] += 1
                if progress_every and done[0] % progress_every == 0:
                    print(f"  [api] {done[0]}/{len(jobs)} done", flush=True)

    executor = ThreadPoolExecutor(max_workers=total_workers, thread_name_prefix="api-pool")
    futures = [executor.submit(task) for _ in range(total_workers)]
    try:
        with _cancel_api_retries_on_sigint(cancel_event):
            for future in as_completed(futures):
                future.result()
    except KeyboardInterrupt:
        cancel_event.set()
        with state_changed:
            state_changed.notify_all()
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    except Exception:
        cancel_event.set()
        with state_changed:
            state_changed.notify_all()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)

    if cancel_event.is_set():
        raise ExecutorInterrupted(
            f"API retries were interrupted; checkpoint contains {len(results)}/{len(jobs)} completed jobs")
    if len(results) != len(jobs):
        raise ExecutorAPIError(
            f"API workers completed only {len(results)}/{len(jobs)} jobs")
    return results


# --------------------------- 单卡 worker ---------------------------
def _worker(jobs_path, out_path, enable_thinking):
    from vllm import LLM, SamplingParams

    # 断点续跑：跳过已完成 id
    done = set()
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass

    jobs = []
    with open(jobs_path, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            if j["id"] not in done:
                jobs.append(j)
    if not jobs:
        return

    llm = LLM(
        model=C.EXECUTOR_MODEL,
        trust_remote_code=True,
        dtype=C.EXECUTOR_DTYPE,
        max_model_len=C.MAX_MODEL_LEN,
        limit_mm_per_prompt={"image": C.MAX_IMAGES},   # 关键：默认仅 1 图
        allowed_local_media_path=C.IMAGE_BASE,          # 关键：允许 file:// 本地图
        # 本机驱动 570 无法加载 vllm 自带 fa2 内核（PTX 版本过高）：ViT 走 SDPA，
        # LM 注意力走 TRITON_ATTN（本机缺 g++，FlashInfer JIT 不可用；与 src/env.sh 一致）
        mm_encoder_attn_backend=C.EXECUTOR_VIT_ATTENTION_BACKEND,
        attention_backend=C.EXECUTOR_ATTENTION_BACKEND,
        gpu_memory_utilization=C.EXECUTOR_GPU_MEMORY_UTILIZATION,
        enforce_eager=False,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=C.GEN_MAX_TOKENS,
                        frequency_penalty=C.GEN_FREQ_PENALTY)
    chat_kwargs = {}
    # Qwen3.5 思维开关：默认关（评测求稳），需要时开
    chat_kwargs["chat_template_kwargs"] = {"enable_thinking": bool(enable_thinking)}

    messages = [D.job_to_messages(j) for j in jobs]
    outputs = llm.chat(messages, sp, **chat_kwargs)

    with open(out_path, "a", encoding="utf-8") as fout:
        for j, o in zip(jobs, outputs):
            raw = o.outputs[0].text
            ans = D.parse_answer(raw, j["valid_letters"])
            op_seq = D.parse_op_sequence(raw)
            fout.write(json.dumps({
                "id": j["id"], "answer": ans, "op_seq": op_seq, "raw": raw,
            }, ensure_ascii=False) + "\n")
            fout.flush()


# --------------------------- 多卡调度 ---------------------------
def run_executor(jobs, gpus=None, tag="exec", enable_thinking=False, workdir=None,
                 use_service="auto", result_sink=None):
    """推理调度：常规部署使用匹配的常驻服务。

    use_service: 'auto'|True|False。返回 {id: {answer, op_seq, raw}}。
    EXEC_BACKEND=api 时改走云 API（闭源模型），忽略本地服务与 gpus。
    默认不重载子进程模型，以免服务故障时悄悄改变部署形态；显式设置
    EXECUTOR_ALLOW_SUBPROCESS_FALLBACK=1 或 use_service=False 才允许该调试路径。
    """
    if not jobs:
        return {}
    if C.EXEC_BACKEND == "api":
        return run_executor_api(jobs, result_sink=result_sink)
    if use_service in ("auto", True):
        eps = service_endpoints()
        if eps:
            matching = _wait_for_matching_service()
            if matching:
                return run_executor_service(jobs, matching, enable_thinking=enable_thinking)
        if use_service is True or not C.EXECUTOR_ALLOW_SUBPROCESS_FALLBACK:
            raise RuntimeError(
                f"常驻执行器 {C.SERVED_MODEL_NAME!r} 未就绪或端点不健康；"
                "请先运行 bash src/serve_vllm.sh。"
                "如需临时子进程回退，请显式设置 EXECUTOR_ALLOW_SUBPROCESS_FALLBACK=1。")
    return _run_executor_subprocess(jobs, gpus=gpus, tag=tag,
                                    enable_thinking=enable_thinking, workdir=workdir)


def _run_executor_subprocess(jobs, gpus=None, tag="exec", enable_thinking=False, workdir=None):
    """子进程多卡回退：每卡一个 vllm 子进程处理一个分片（会重载模型）。"""
    gpus = gpus or C.EXECUTOR_GPUS
    workdir = workdir or os.path.join(C.CURRENT_RUN_DIR, "_exec_tmp", tag)
    os.makedirs(workdir, exist_ok=True)

    n = len(gpus)
    shards = [[] for _ in range(n)]
    for i, j in enumerate(jobs):
        shards[i % n].append(j)

    procs = []
    out_paths = []
    for gi, gpu in enumerate(gpus):
        if not shards[gi]:
            continue
        jp = os.path.join(workdir, f"jobs_g{gpu}.jsonl")
        op = os.path.join(workdir, f"out_g{gpu}.jsonl")
        with open(jp, "w", encoding="utf-8") as f:
            for j in shards[gi]:
                f.write(json.dumps(j, ensure_ascii=False) + "\n")
        out_paths.append(op)
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        cmd = [sys.executable, "-m", "skillevo.executor", "--worker",
               "--jobs", jp, "--out", op]
        if enable_thinking:
            cmd.append("--thinking")
        log = open(os.path.join(workdir, f"log_g{gpu}.txt"), "w")
        procs.append(subprocess.Popen(cmd, env=env, cwd=C.ROOT,
                                      stdout=log, stderr=subprocess.STDOUT))
    for p in procs:
        p.wait()

    results = {}
    for op in out_paths:
        if not os.path.exists(op):
            continue
        with open(op, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                results[r["id"]] = r
    return results


def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--jobs")
    ap.add_argument("--out")
    ap.add_argument("--thinking", action="store_true")
    args = ap.parse_args()
    if args.worker:
        _worker(args.jobs, args.out, args.thinking)
    else:
        print("run as library via run_executor(), or with --worker", file=sys.stderr)


if __name__ == "__main__":
    _cli()
