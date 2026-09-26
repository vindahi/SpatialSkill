#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据加载、图像路径解析、MCQ prompt 构建、执行器输出解析（选项 + op 序列）。"""
import json
import os
import re

from . import config as C


# --------------------------- 加载 ---------------------------
def load_records(split, limit=0):
    path = C.TRAIN_JSONL if split == "train" else C.VAL_JSONL
    recs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            recs.append(json.loads(line))
    if limit:
        recs = recs[:limit]
    return recs


def resolve_images(rec):
    """把 images 相对路径解析为绝对路径，限制最多 MAX_IMAGES 张。"""
    paths = []
    for rel in rec.get("images", [])[: C.MAX_IMAGES]:
        p = os.path.join(C.IMAGE_BASE, rel)
        if os.path.exists(p):
            paths.append(p)
    return paths


# --------------------------- prompt ---------------------------
_LETTERS = ["A", "B", "C", "D", "E", "F"]

BASE_INSTR = (
    "You are answering a cross-view urban spatial-reasoning multiple-choice question. "
    "Multiple images are given as different views of the same scene."
)


def _choices_block(choices):
    return "\n".join(f"  {k}. {v}" for k, v in choices.items())


def build_question_text(rec, skill_text=None, cot=True, evidence=False):
    """构建题目文本部分（图像单独传）。skill_text 为注入的技能指引（可空）。

    evidence 仅在 cot=False 时生效：直答优先（ANSWER 行在前，保持与部署形态一致），
    再追加 1-2 句关键视觉证据，供进化反思归因用（反思器看不到图）。"""
    parts = [BASE_INSTR]
    if skill_text:
        parts.append(
            "\n[Cross-View Urban Spatial-Reasoning Skill Manual] The following are "
            "reusable, image-agnostic reasoning strategies distilled from past experience. "
            "Apply the relevant ones; they are general procedures, never answers:\n"
            + skill_text)
    parts.append(f"\nQuestion: {rec['question']}")
    parts.append("Choices:\n" + _choices_block(rec["choices"]))
    if cot:
        parts.append(
            "\nThink step by step briefly, then end with a final line EXACTLY like:\n"
            "ANSWER: <letter>")
    elif evidence:
        parts.append(
            "\nAnswer FIRST with one line EXACTLY like:\nANSWER: <letter>\n"
            "Then add 1-2 short sentences naming the key visual evidence you used.")
    else:
        parts.append("\nRespond with EXACTLY one line:\nANSWER: <letter>")
    return "\n".join(parts)


def build_job(rec, skill_text=None, cot=True, evidence=False):
    """构建可序列化的推理任务（供多卡子进程分片）。图像存绝对路径。"""
    return {
        "id": rec["id"],
        "images": resolve_images(rec),
        "text": build_question_text(rec, skill_text, cot, evidence),
        "valid_letters": list(rec["choices"].keys()),
    }


def job_to_messages(job):
    """job -> openai 风格多模态 messages（图像用 file:// 本地 URL）。"""
    content = [{"type": "image_url", "image_url": {"url": "file://" + p}}
               for p in job["images"]]
    content.append({"type": "text", "text": job["text"]})
    return [{"role": "user", "content": content}]


def image_b64(path):
    """本地图片 -> (mime, base64 字符串)。云 API 无法访问 file:// 路径，需内联传图。"""
    import base64
    import mimetypes
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as f:
        return mime, base64.b64encode(f.read()).decode()


def job_to_messages_api(job):
    """job -> openai 风格多模态 messages（图像内联 base64 data URL，供云 API 用）。"""
    content = []
    for p in job["images"]:
        mime, b64 = image_b64(p)
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"}})
    content.append({"type": "text", "text": job["text"]})
    return [{"role": "user", "content": content}]


# --------------------------- 解析 ---------------------------
def parse_answer(text, valid_letters):
    """从执行器输出解析选项字母：优先 ANSWER: 之后，回退全文最后一个合法字母。"""
    valid = set(valid_letters)
    m = list(re.finditer(r"ANSWER\s*[:：]\s*([A-F])", text, re.IGNORECASE))
    if m:
        c = m[-1].group(1).upper()
        if c in valid:
            return c
    # 回退：全文最后一个孤立的合法字母
    cand = None
    for mm in re.finditer(r"(?<![A-Za-z])([A-F])(?![A-Za-z])", text.upper()):
        if mm.group(1) in valid:
            cand = mm.group(1)
    return cand


def parse_op_sequence(text):
    """从执行器 CoT 里粗解析走过的 op 序列（用于 ExecutionLapse 检测）。

    不依赖模型输出 JSON：用关键词匹配把自由文本 CoT 映射到 OPS。返回有序去重的 op 列表。
    """
    t = text.lower()
    hit = []
    kw = {
        "locate_landmark": ["identify", "locate", "find the", "recognize", "spot the"],
        "cross_view_match": ["match", "same object", "corresponds", "co-visible", "across views", "in view"],
        "estimate_direction": ["direction", "north", "south", "east", "west", "left of", "right of", "facing"],
        "estimate_distance": ["distance", "closer", "farther", "nearer", "meters", "far from"],
        "spatial_relation": ["relationship", "relative position", "between", "next to", "adjacent"],
        "camera_motion": ["camera move", "viewpoint move", "rotate", "moved from", "translation"],
        "perspective_transform": ["from the perspective", "stand at", "if you turn", "point of view", "imagine"],
        "counting": ["count", "how many", "number of"],
        "world_knowledge": ["typically", "usually", "commonly", "function", "type of"],
        "eliminate_option": ["eliminate", "rule out", "cannot be", "option a", "option b", "not correct"],
        "final_decision": ["answer:", "therefore", "final", "the correct option", "choose"],
    }
    for op, keys in kw.items():
        idx = min([t.find(k) for k in keys if t.find(k) >= 0] or [1e9])
        if idx < 1e9:
            hit.append((idx, op))
    hit.sort()
    seq, seen = [], set()
    for _, op in hit:
        if op not in seen:
            seq.append(op)
            seen.add(op)
    return seq


def gold_op_sequence(rec):
    """记录里 L3 蒸馏好的 op 序列（去重保序），作为该题的 op 先验。"""
    steps = rec.get("reasoning_steps") or []
    seq, seen = [], set()
    for s in steps:
        op = s.get("op")
        if op and op not in seen:
            seq.append(op)
            seen.add(op)
    return seq
