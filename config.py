#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全局配置：路径、模型、维度常量、超参默认值。所有模块从这里取常量。"""
import datetime as _dt
import ast
import json
import os
import re

# --------------------------- 路径 ---------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_deployment_defaults():
    """加载统一部署配置，但不覆盖外部环境变量。

    ``src/env.sh`` 负责 shell 启动脚本；Python 直接启动时由这里加载同一
    个简单 KEY=value 文件，避免模型和地址在两套配置中漂移。
    """
    path = os.environ.get(
        "SPATIALSKILL_DEPLOYMENT_CONFIG",
        os.path.join(ROOT, "config", "deployment.env"),
    )
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key.startswith("export "):
                key = key[7:].strip()
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] == '"':
                try:
                    value = ast.literal_eval(value)
                except (SyntaxError, ValueError):
                    value = value[1:-1]
            os.environ.setdefault(key, value)


_load_deployment_defaults()
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")  # 绕开 flashinfer JIT
# 注意：vllm 0.19 无 VLLM_ATTENTION_BACKEND 环境变量；本机驱动 570 与自带 fa2 内核
# PTX 不兼容，注意力后端须显式传参（serve 脚本 --attention-backend / LLM() attention_backend）

DATA_DIR = os.path.join(ROOT, "dataset", "data")
IMAGE_BASE = os.path.join(ROOT, "CityCube-Bench")   # images 字段相对此目录
MODEL_DIR = os.path.join(ROOT, "model")
RUNS_DIR = os.path.join(ROOT, "runs")
LOGS_DIR = os.path.join(ROOT, "logs")

TRAIN_JSONL = os.path.join(DATA_DIR, "spatialskill_train.jsonl")
VAL_JSONL = os.path.join(DATA_DIR, "spatialskill_val.jsonl")

# --------------------------- 模型 ---------------------------
# 本机权重路径与执行器名称统一由 config/deployment.env 提供；环境变量可临时覆盖。
EXECUTOR_MODEL_NAME = os.environ.get(
    "EXEC_MODEL_NAME",
    os.environ.get("EXECUTOR_MODEL_NAME", "qwen3_5_4b"))
SERVED_MODEL_NAME = EXECUTOR_MODEL_NAME

# EXEC_MODEL_PATH 可覆盖当前执行器；未设置时使用部署配置中的路径。
EXECUTOR_MODEL = (
    os.environ.get("EXEC_MODEL_PATH")
    or os.environ.get("EXECUTOR_MODEL_PATH")
    or os.path.join(MODEL_DIR, EXECUTOR_MODEL_NAME)
)
# tokenizer 折算与执行器使用同一份当前权重路径。
MODEL_PATHS = {SERVED_MODEL_NAME: EXECUTOR_MODEL}

# 技能进化反思器模型；BRAIN_MODEL_PATH 是旧变量名，保留为兼容覆盖入口。
REFLECTOR_MODEL = os.environ.get(
    "REFLECTOR_MODEL_PATH",
    os.environ.get("BRAIN_MODEL_PATH", EXECUTOR_MODEL))
BRAIN_MODEL = REFLECTOR_MODEL
EMB_MODEL = os.path.join(MODEL_DIR, "bge_m3")            # 检索嵌入（未迁移，当前未使用）
EMB_MODEL_FALLBACK = os.path.join(MODEL_DIR, "all_minilm")  #（未迁移，当前未使用）

# --------------------------- 维度 ---------------------------
CVSI_LABELS = ["SR", "PT", "MR", "WK", "CR"]
REL_LABELS = ["obj-obj", "cam-obj", "cam-cam"]
VIEWPOINT_LABELS = ["rotation", "hover", "scale_translation", "high_low"]

# L3 的 12 个操作原语（与 dataset/build/03_decompose_cot.py 保持一致）
OPS = [
    "locate_landmark", "cross_view_match", "estimate_direction", "estimate_distance",
    "spatial_relation", "camera_motion", "perspective_transform", "counting",
    "world_knowledge", "eliminate_option", "final_decision", "other",
]

# 反思类型（严格对齐 EmbodiSkill 原文：成功 S_NEW/S_BETTER，失败 FAIL_SKILL/FAIL_EXECUTION）
REFLECT_TYPES = ["S_NEW", "S_BETTER", "FAIL_SKILL", "FAIL_EXECUTION"]

# --------------------------- 技能手册（自然语言 manual，对齐 EmbodiSkill 本体） ---------------------------
# 4 个固定板块（把 ALFWorld 的 Planning/Search/Execution/Failure Avoidance 域适配到跨视角空间推理）
MANUAL_SECTION_TITLES = ["Planning", "Cross-View Alignment", "Spatial Reasoning", "Failure Avoidance"]
MANUAL_DISPLAY_NAME = "Cross-View Urban Spatial-Reasoning Skill Manual"
# refactor 触发阈值（板块条目数 / 执行提醒数超限时 LLM 精简）
# 阈值 36 / 压缩目标 28 / 每窗净增预算 3：三者拉开缓冲带，避免“涨到 35 压到 20
# 一窗又涨回”的锯齿震荡（旧参数 30/20 导致 44/45 窗都在 refactor，手册翻修致死）
MANUAL_SECTIONS_REFACTOR_THRESHOLD = 36
EXECUTION_NOTES_REFACTOR_THRESHOLD = 25

# --------------------------- 稳定进化（E4v2：反 churn 三机制） ---------------------------
EVOLVE_SNEW_BUDGET = 3     # 每窗净增规则上限（防 S_NEW 洪泛，实测峰值 33 条/窗）
PROTECT_STREAK = 3         # 规则连续存活窗数达标即受保护（refactor/截断不得动）
EVOLVE_PROBE_N = 200       # 探针集大小（从 train 划出，不进进化流，裁决修订接受/拒绝）
EVOLVE_PROBE_EPS = 1.5     # 棘轮容差（百分点）：候选手册探针分 >= 基准 - EPS 才接受
EVOLVE_REJECT_REFRESH = 3  # 连续拒绝 N 次后重测基准分，防幸运高分基准造成永久卡死
EVOLVE_VAL_EVERY = 5       # 每 N 窗跑一次 val 监控（仅记录进 evolve_log，不参与选择）
# --------------------------- 探针驱动的二维路由自动搜索（第二步） ---------------------------
POLICY_STRATIFIED_PROBE = True  # 探针按 val 五类比例分层抽样（与 full val 分布一致）
POLICY_SEARCH_MARGIN = 1.5      # 防过拟合守卫①：非默认格子须高出默认格 >= 此百分点才切换
POLICY_MIN_PROBE = 10           # 防过拟合守卫②：类别探针样本 < 此数退回默认格子
# LLM token 预算
REFLECTION_MAX_TOKENS = 1024
REVISION_MAX_TOKENS = 8192  # 板块修订 + refactor 都需大空间，统一 8192 防截断
REFACTOR_MAX_TOKENS = 8192  # refactor 输出全册（4板块），需更大空间防截断
# 窗口内反思并发（库冻结，安全并行）。反思器为 Qwen3-8B 单卡服务。
LLM_CONCURRENCY = int(os.environ.get("SKILL_LLM_CONCURRENCY", "16"))

# --------------------------- 进化超参默认值 ---------------------------
K_MAX_REFLECT = 1        # 每轨迹最多反思数
N_STAGES = 10            # 修订阶段数
B_INTERVAL = 100         # 修订间隔（缓冲反思数达到即修订）
BODY_BUDGET = 15         # 每分片 body 软上限
APP_BUDGET = 10          # 每分片 appendix 软上限
TOPK_BODY = 4            # 检索注入 body 条数

# 检索打分权重
RETRIEVAL_ALPHA = 0.5    # op 先验重合（Jaccard）
RETRIEVAL_BETA = 0.4     # 问题嵌入余弦
RETRIEVAL_GAMMA = 0.1    # 历史 win_rate

# --------------------------- 技能进化反思器 LLM（驱动反思/修订/重构） ---------------------------
# 后端三选一：
#   "siliconflow" 硅基流动云 API（OpenAI 兼容，默认，需 SILICONFLOW_API_KEY，DeepSeek-V3.2）
#   "vllm"        本地常驻 vLLM（OpenAI 兼容，需 SKILL_LLM_URL，Qwen3-8B @ GPU0）
#   "deepseek"    DeepSeek 官方云 API（需 DEEPSEEK_API_KEY）
SKILL_LLM_BACKEND = os.environ.get(
    "SKILL_LLM_BACKEND",
    os.environ.get("REFLECTOR_BACKEND", "vllm"),
).strip().lower()
# 本地 vLLM 反思器端点（chat completions）与服务名（仅 backend=vllm 时用）
REFLECTOR_HOST = os.environ.get("REFLECTOR_HOST", "127.0.0.1")
REFLECTOR_PORT = int(os.environ.get("REFLECTOR_PORT", "8001"))


def _csv_texts(value):
    return tuple(item.strip() for item in str(value or "").split(",") if item.strip())


REFLECTOR_PORTS = _csv_texts(os.environ.get("REFLECTOR_PORTS", "")) or (str(REFLECTOR_PORT),)
REFLECTOR_REPLICAS = int(os.environ.get("REFLECTOR_REPLICAS", str(len(REFLECTOR_PORTS))))
REFLECTOR_MAX_NUM_SEQS = int(os.environ.get("REFLECTOR_MAX_NUM_SEQS", "4"))

# Multiple local replicas are exposed as independent OpenAI-compatible
# endpoints. An explicit SKILL_LLM_URLS override wins; SKILL_LLM_URL remains
# the first endpoint for legacy callers and diagnostics.
_skill_llm_urls = _csv_texts(os.environ.get("SKILL_LLM_URLS", ""))
if not _skill_llm_urls:
    _skill_llm_url_override = os.environ.get("SKILL_LLM_URL", "").strip()
    _skill_llm_urls = ((_skill_llm_url_override,) if _skill_llm_url_override else tuple(
        f"http://{REFLECTOR_HOST}:{port}/v1/chat/completions" for port in REFLECTOR_PORTS))
SKILL_LLM_URLS = _skill_llm_urls
SKILL_LLM_URL = os.environ.get(
    "SKILL_LLM_URL",
    SKILL_LLM_URLS[0],
)
SKILL_LLM_MODEL = os.environ.get(
    "SKILL_LLM_MODEL",
    os.environ.get("REFLECTOR_MODEL_NAME", "qwen3_8b_reflector"),
)

# --------------------------- 硅基流动 SiliconFlow（默认云后端，backend=siliconflow 时用） ---------------------------
# API key 从环境变量 SILICONFLOW_API_KEY 读；url/model 来自统一部署配置。
SF_URL = os.environ.get(
    "SILICONFLOW_URL",
    os.environ.get("REFLECTOR_SILICONFLOW_URL", "https://api.siliconflow.cn/v1/chat/completions"),
)
SF_MODEL = os.environ.get(
    "SILICONFLOW_MODEL",
    os.environ.get("REFLECTOR_SILICONFLOW_MODEL", "deepseek-ai/DeepSeek-V3.2"),
)


def _env_flag(name, default):
    """读取布尔环境变量；未设置时使用部署配置给出的默认值。"""
    value = os.environ.get(name, default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


# Qwen3.5 的思维链会占用 max_tokens，但反思器只需要最终 JSON。
# SiliconFlow 将该开关作为 chat-completions 的顶层请求字段。
SF_ENABLE_THINKING = _env_flag(
    "SILICONFLOW_ENABLE_THINKING",
    os.environ.get("REFLECTOR_SILICONFLOW_ENABLE_THINKING", "0"),
)

# --------------------------- DeepSeek（可选后端，仅 SKILL_LLM_BACKEND=deepseek 时用） ---------------------------
# API key 从环境变量 DEEPSEEK_API_KEY 读；url/model 来自统一部署配置。
DS_URL = os.environ.get(
    "DEEPSEEK_URL",
    os.environ.get("REFLECTOR_DEEPSEEK_URL", "https://api.deepseek.com/chat/completions"),
)
DS_MODEL = os.environ.get(
    "DEEPSEEK_MODEL",
    os.environ.get("REFLECTOR_DEEPSEEK_MODEL", "deepseek-chat"),
)
DS_TEMPERATURE = 0.2
# 云后端读超时（秒）。DeepSeek-V3.2 带推理 + 修订大 JSON（4096 token）较慢，
# 120s 会触发 ReadTimeout 导致板块修订被跳过；给足 300s，可用 SKILL_LLM_TIMEOUT 覆盖。
DS_TIMEOUT = int(os.environ.get(
    "SKILL_LLM_TIMEOUT",
    os.environ.get("REFLECTOR_TIMEOUT", "300"),
))
DS_RETRIES = 3

# --------------------------- 推理默认值 ---------------------------
MAX_MODEL_LEN = int(os.environ.get(
    "EXECUTOR_SERVICE_MAX_MODEL_LEN", "32768"))  # 多图占 token 多，给足（最多 10 图）
MAX_IMAGES = 10          # 单题最多图数（数据里最高 10）
# skill 模式注入手册后 CoT 显著变长，1024 会截断 16% 回答（无 ANSWER 行），给足 3072
GEN_MAX_TOKENS = 3072
# 重复惩罚对本任务有害：方向/运动推理需反复用同一批方位词（cot 基线 53.2→49.2，MR 63→49），置 0 禁用；
# 枚举死循环改由二段式 ANSWER 兜底处理
GEN_FREQ_PENALTY = 0.0
# 二段式兜底：首次输出无 ANSWER 行时，追加一次短请求强制出答案
ANSWER_RETRY_MAX_TOKENS = 16

# --------------------------- 常驻 vLLM 执行器服务 ---------------------------
# 执行器常驻服务，客户端并发轮询。当前模型、GPU 和副本数均由 deployment.env 定义；
# 临时切换模型时需同步修改名称、权重路径并用 serve_vllm.sh 重新部署。
SERVICE_DIR = os.path.join(RUNS_DIR, "_service")
SERVICE_FILE = os.path.join(SERVICE_DIR, "endpoints.json")
SERVICE_CONCURRENCY = int(os.environ.get("EXEC_SERVICE_CONCURRENCY", "48"))  # 每副本并发请求数
EXECUTOR_SERVICE_HOST = os.environ.get("EXECUTOR_SERVICE_HOST", "127.0.0.1")


def _csv_ints(value, default):
    try:
        values = [int(x.strip()) for x in value.split(",") if x.strip()]
    except (AttributeError, ValueError):
        values = []
    return values or list(default)


EXECUTOR_GPUS = _csv_ints(
    os.environ.get("EXECUTOR_SERVICE_GPUS", "2,3"), (2, 3))
EXECUTOR_PORTS = _csv_ints(
    os.environ.get("EXECUTOR_SERVICE_PORTS", "8101,8102"), (8101, 8102))
EXECUTOR_DTYPE = os.environ.get("EXECUTOR_SERVICE_DTYPE", "bfloat16")
EXECUTOR_MAX_MODEL_LEN = int(os.environ.get("EXECUTOR_SERVICE_MAX_MODEL_LEN", "32768"))
EXECUTOR_GPU_MEMORY_UTILIZATION = float(os.environ.get(
    "EXECUTOR_SERVICE_GPU_MEMORY_UTILIZATION", "0.95"))
EXECUTOR_SERVICE_WAIT_SECONDS = int(os.environ.get(
    "EXECUTOR_SERVICE_WAIT_SECONDS", "180"))
EXECUTOR_ALLOW_SUBPROCESS_FALLBACK = os.environ.get(
    "EXECUTOR_ALLOW_SUBPROCESS_FALLBACK", "0",
).strip().lower() in {"1", "true", "yes", "on"}
EXECUTOR_ATTENTION_BACKEND = os.environ.get(
    "ATTN_BACKEND",
    os.environ.get("VLLM_ATTENTION_BACKEND", "TRITON_ATTN"),
)
EXECUTOR_VIT_ATTENTION_BACKEND = os.environ.get(
    "VIT_ATTN",
    os.environ.get("VLLM_VIT_ATTENTION_BACKEND", "TORCH_SDPA"),
)

# --------------------------- 执行器后端（本地 vLLM 服务 / 云 API 池） ---------------------------
# EXEC_BACKEND=service（默认）：本地 vLLM 常驻服务
# EXEC_BACKEND=api          ：JSON 配置的外部 API 池，见 executor.run_executor_api
EXEC_BACKEND = os.environ.get(
    "EXEC_BACKEND",
    os.environ.get("EXECUTOR_BACKEND", "service"),
).strip().lower()
EXEC_API_CONFIG = os.environ.get(
    "EXEC_API_CONFIG",
    os.environ.get("EXECUTOR_API_CONFIG", "config/executor_apis.json"),
).strip()
EXEC_API_RUN_NAME = os.environ.get(
    "EXEC_API_RUN_NAME",
    os.environ.get("EXECUTOR_API_RUN_NAME", "api_pool"),
).strip() or "api_pool"


def _executor_api_config_path(value):
    if not value:
        return ""
    return value if os.path.isabs(value) else os.path.join(ROOT, value)


def _api_config_int(value, field, api_name, minimum=0):
    if isinstance(value, bool):
        raise ValueError(f"API {api_name!r} 的 {field} 必须是整数")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"API {api_name!r} 的 {field} 必须是整数") from e
    if parsed < minimum:
        raise ValueError(f"API {api_name!r} 的 {field} 必须 >= {minimum}")
    return parsed


def _api_config_text(value, field, api_name):
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"API {api_name!r} 缺少必填字段 {field}")
    return text


def _expand_api_config(value):
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, dict):
        return {key: _expand_api_config(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_api_config(item) for item in value]
    return value


def _load_executor_api_pools(path):
    """读取外部执行器 API 池。

    JSON 的每个 API 都通过 ``api_key_env`` 指向密钥环境变量，避免把密钥写入仓库。
    单个 API 的并发彼此独立；执行器调度时总并发为所有 ``concurrency`` 之和。
    连续失败达到 ``failure_threshold`` 后，URL 会冻结 ``cooldown_seconds``；
    冷却结束时仅允许一个探测请求恢复该 URL。可用正数
    ``max_freeze_cycles`` 在多次恢复失败后永久移出 URL；0 表示持续探测。
    """
    if not path:
        return ()
    if not os.path.isfile(path):
        # Delay the actionable error until the API backend is actually used;
        # importing skillevo.config must remain possible for local tooling.
        return ()
    try:
        with open(path, encoding="utf-8") as fh:
            document = json.load(fh)
    except json.JSONDecodeError as e:
        raise ValueError(f"执行器 API JSON 无法解析：{path}: {e}") from e
    if not isinstance(document, dict):
        raise ValueError(f"执行器 API JSON 顶层必须是对象：{path}")
    document = _expand_api_config(document)
    defaults = document.get("defaults", {})
    entries = document.get("apis")
    if not isinstance(defaults, dict):
        raise ValueError(f"执行器 API JSON 的 defaults 必须是对象：{path}")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"执行器 API JSON 需要非空 apis 数组：{path}")

    pools = []
    names = set()
    for index, raw_entry in enumerate(entries, start=1):
        if not isinstance(raw_entry, dict):
            raise ValueError(f"执行器 API JSON 的 apis[{index - 1}] 必须是对象")
        entry = dict(defaults)
        entry.update(raw_entry)
        api_name = str(entry.get("name") or f"api-{index}").strip()
        if not api_name:
            raise ValueError(f"执行器 API JSON 的 apis[{index - 1}] name 不能为空")
        if api_name in names:
            raise ValueError(f"执行器 API 名称重复：{api_name!r}")
        names.add(api_name)

        style = _api_config_text(entry.get("style", "openai"), "style", api_name).lower()
        if style not in {"openai", "anthropic", "gemini"}:
            raise ValueError(
                f"API {api_name!r} 的 style={style!r} 非法（openai|anthropic|gemini）")
        url = _api_config_text(entry.get("base_url", entry.get("url")), "base_url", api_name)
        key_env = str(entry.get("api_key_env", "") or "").strip()
        if key_env and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key_env):
            raise ValueError(f"API {api_name!r} 的 api_key_env 非法：{key_env!r}")
        api_key = str(entry.get("api_key", entry.get("key", entry.get("token", ""))) or "")
        if key_env:
            api_key = os.environ.get(key_env, api_key)
        pools.append({
            "name": api_name,
            "style": style,
            "url": url,
            "model": _api_config_text(entry.get("model"), "model", api_name),
            "api_key_env": key_env,
            "api_key": api_key,
            "concurrency": _api_config_int(entry.get("concurrency"), "concurrency", api_name, 1),
            "timeout": _api_config_int(entry.get("timeout", 300), "timeout", api_name, 1),
            "max_retries": _api_config_int(
                entry.get("max_retries", 8), "max_retries", api_name, 0),
            "retry_delay_seconds": _api_config_int(
                entry.get("retry_delay_seconds", 60), "retry_delay_seconds", api_name, 0),
            "failure_threshold": _api_config_int(
                entry.get("failure_threshold", 3), "failure_threshold", api_name, 1),
            "cooldown_seconds": _api_config_int(
                entry.get("cooldown_seconds", 900), "cooldown_seconds", api_name, 0),
            "max_freeze_cycles": _api_config_int(
                entry.get("max_freeze_cycles", 0), "max_freeze_cycles", api_name, 0),
        })
    return tuple(pools)


EXEC_API_CONFIG_PATH = _executor_api_config_path(EXEC_API_CONFIG)
EXEC_API_POOLS = _load_executor_api_pools(EXEC_API_CONFIG_PATH)
# 保留首个模型名供 tokenizer / 历史调用方使用；真正请求参数以每个 API 条目为准。
EXEC_API_MODEL = EXEC_API_POOLS[0]["model"] if EXEC_API_POOLS else ""
# Read-only aliases for older library callers. New code must use
# ``EXEC_API_POOLS`` because these values describe only the first API.
_first_api = EXEC_API_POOLS[0] if EXEC_API_POOLS else {}
EXEC_API_STYLE = _first_api.get("style", "openai")
EXEC_API_URL = _first_api.get("url", "")
EXEC_API_KEY = os.environ.get(
    _first_api.get("api_key_env", ""), _first_api.get("api_key", "")
)
EXEC_API_CONCURRENCY = sum(api["concurrency"] for api in EXEC_API_POOLS)
EXEC_API_TIMEOUT = _first_api.get("timeout", 300)
EXEC_API_MAX_ATTEMPTS = max(1, int(_first_api.get("max_retries", 8))) if _first_api else 8
EXEC_API_RETRY_DELAY_SECONDS = _first_api.get("retry_delay_seconds", 60)


def dim_key(category, relation=None, viewpoint=None):
    """分片键：category 为主，可选 relation/viewpoint 细分。None 用 '*' 占位。"""
    return "|".join([category or "*", relation or "*", viewpoint or "*"])


def ensure_dirs():
    for d in (RUNS_DIR, LOGS_DIR, CURRENT_RUN_DIR, CURRENT_LOG_DIR):
        os.makedirs(d, exist_ok=True)


# --------------------------- 实验产物命名与版本管理 ---------------------------
# 每次独立启动使用 runs/<executor>_<YYYYMMDD>_vNNN/ 与同名 logs 目录。
# 默认按当前本地日期新建上下文；复用已有上下文必须显式 RUN_REUSE=1 + RUN_ID。
# RUN_MODEL_NAME 是 shell 运行后的结果变量；直接 Python 启动时不读取旧的同名环境变量。
if os.environ.get("RUN_CONTEXT_MODEL_EXPLICIT", "0") == "1":
    RUN_MODEL_NAME = os.environ.get("RUN_CONTEXT_MODEL", SERVED_MODEL_NAME)
elif EXEC_BACKEND == "api":
    # API 池没有单一权重名；用显式运行名隔离整个池的产物。
    RUN_MODEL_NAME = EXEC_API_RUN_NAME
else:
    RUN_MODEL_NAME = SERVED_MODEL_NAME


def _safe_model_name(name):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(name or "executor")).strip("_").lower() or "executor"


def _next_run_id(model_name=None, run_date=None):
    safe = _safe_model_name(model_name or RUN_MODEL_NAME)
    day = run_date or RUN_DATE
    prefix = f"{safe}_{day}_v"
    highest = 0
    if os.path.isdir(RUNS_DIR):
        for entry in os.listdir(RUNS_DIR):
            if not entry.startswith(prefix):
                continue
            suffix = entry[len(prefix):]
            if suffix.isdigit():
                highest = max(highest, int(suffix))
    return f"{prefix}{highest + 1:03d}"


def _run_id_parts(run_id):
    """Extract model prefix and YYYYMMDD from a reusable run id."""
    match = re.fullmatch(r"(.+)_(\d{8})_v\d+", str(run_id or ""))
    if not match:
        raise ValueError(
            f"invalid RUN_ID (expected ..._YYYYMMDD_vNNN): {run_id!r}")
    return match.group(1), match.group(2)


_env_run_id = os.environ.get("RUN_ID", "")
_reuse_context = (
    os.environ.get("RUN_REUSE", "0") == "1"
    or os.environ.get("RUN_CONTEXT_INITIALIZED", "0") == "1"
)
if _reuse_context:
    if not _env_run_id:
        raise RuntimeError("RUN_REUSE=1 requires RUN_ID=...")
    RUN_ID = _env_run_id
    _run_model, RUN_DATE = _run_id_parts(RUN_ID)
    if _run_model != _safe_model_name(RUN_MODEL_NAME):
        raise RuntimeError(
            f"RUN_ID model={_run_model!r} does not match "
            f"current model={_safe_model_name(RUN_MODEL_NAME)!r}")
else:
    # Ignore inherited RUN_DATE/RUN_ID here.  Direct Python launches should have
    # the same fresh-context behavior as the shell launchers in src/env.sh.
    RUN_DATE = _dt.datetime.now().strftime("%Y%m%d")
    RUN_ID = _next_run_id(run_date=RUN_DATE)
CURRENT_RUN_DIR = os.path.join(RUNS_DIR, RUN_ID)
CURRENT_LOG_DIR = os.path.join(LOGS_DIR, RUN_ID)


def run_path(*parts):
    """返回当前运行目录下的产物路径。"""
    return os.path.join(CURRENT_RUN_DIR, *parts)


def log_path(*parts):
    """返回当前运行目录下的日志路径。"""
    return os.path.join(CURRENT_LOG_DIR, *parts)


def resolve_run_output(name):
    """解析输出目录：默认放当前运行目录，也兼容显式绝对/旧 runs/ 路径。"""
    if os.path.isabs(name):
        return name
    normalized = os.path.normpath(name)
    if normalized == "runs" or normalized.startswith("runs" + os.sep):
        return os.path.join(ROOT, normalized)
    return run_path(normalized)
