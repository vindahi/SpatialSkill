#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import json
import os
import re
from datetime import datetime, timezone

from . import config as C
from .prompts import SECTION_TITLES, MANUAL_DISPLAY_NAME


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _norm_key(text):
    s = re.sub(r"\s+", " ", str(text or "").strip())
    s = s.rstrip(" .;:,_-")
    return s.casefold()


def item_key(text):
    """公共规则文本归一化 key（供进化层做存活追踪/预算匹配，与去重一致）。"""
    return _norm_key(text)


def _title_key(title):
    return re.sub(r"[^a-z]+", " ", str(title or "").strip().lower()).strip()


_ALLOWED_TITLE_MAP = {_title_key(t): t for t in SECTION_TITLES}


def empty_state():
    return {
        "version": 1,
        "updated_at_utc": _utc_now(),
        "sections": [{"title": t, "items": []} for t in SECTION_TITLES],
        "execution_notes": [],
    }


def _dedupe_preserve_order(items):
    seen, out = set(), []
    for it in items:
        text = re.sub(r"\s+", " ", str(it).strip())
        if not text:
            continue
        k = _norm_key(text)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(text)
    return out


class Manual:
    def __init__(self, state=None):
        self.state = normalize_state(state) if state else empty_state()

    # ------------------------------- 读 -------------------------------
    @property
    def version(self):
        return int(self.state.get("version", 1))

    def sections(self):
        return self.state.get("sections", [])

    def execution_notes(self):
        return self.state.get("execution_notes", [])

    def skill_to_section(self):
        m = {}
        for sec in self.sections():
            title = sec.get("title", "")
            for it in sec.get("items", []):
                t = str(it).strip()
                if t and t not in m:
                    m[t] = title
        return m

    def n_items(self):
        return sum(len([i for i in s.get("items", []) if str(i).strip()])
                   for s in self.sections())

    def n_notes(self):
        return len(self.execution_notes())

    def is_empty(self):
        return self.n_items() == 0 and self.n_notes() == 0

    def summary(self):
        return {"version": self.version, "n_items": self.n_items(),
                "n_notes": self.n_notes(),
                "by_section": {s["title"]: len(s["items"]) for s in self.sections()}}

    # ------------------------------- 渲染 -------------------------------
    def sections_text(self):
        parts = [f"{MANUAL_DISPLAY_NAME} v{self.version}"]
        for sec in self.sections():
            title = str(sec.get("title", "")).strip()
            items = [str(i).strip() for i in sec.get("items", []) if str(i).strip()]
            if title:
                parts.append(f"\n[{title}]")
            for idx, it in enumerate(items, 1):
                parts.append(f"{idx}. {it}")
        return "\n".join(parts).strip()

    def notes_text(self):
        notes = self.execution_notes()
        if not notes:
            return "[Execution Notes Appendix]\n(none)"
        parts = ["[Execution Notes Appendix]"]
        for idx, n in enumerate(notes, 1):
            parts.append(f"{idx}. Target Section: {n.get('target_section', '(none)')}")
            parts.append(f"   Target Skill: {n.get('target_skill', '(none)')}")
            parts.append(f"   Note: {n.get('note', '')}")
        return "\n".join(parts).strip()

    def render_text(self, include_notes=True):
        parts = [self.sections_text()]
        if include_notes and self.execution_notes():
            parts.append(self.notes_text())
        return "\n".join(parts).strip()

    def prompt_lines(self, include_notes=True):
        lines = []
        for sec in self.sections():
            title = str(sec.get("title", "")).strip()
            for it in sec.get("items", []):
                t = str(it).strip()
                if t:
                    lines.append(f"[{title}] {t}" if title else t)
        if include_notes:
            for n in self.execution_notes():
                note = str(n.get("note", "")).strip()
                if not note:
                    continue
                sk = str(n.get("target_skill", "")).strip()
                sec = str(n.get("target_section", "")).strip()
                if sk:
                    label = f"{sec}: {sk}" if sec else sk
                    lines.append(f"[Execution Note for {label}] {note}")
                else:
                    lines.append(f"[Execution Note] {note}")
        return lines

    # ------------------------------- 持久化 -------------------------------
    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.state["updated_at_utc"] = _utc_now()
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))


def _normalize_notes(notes, skill_to_section=None):
    out, idx_by_key = [], {}
    for raw in (notes if isinstance(notes, list) else []):
        if not isinstance(raw, dict):
            continue
        note = re.sub(r"\s+", " ", str(raw.get("note", "")).strip())
        ts = str(raw.get("target_section", "")).strip()
        tk = str(raw.get("target_skill", "")).strip()
        if not note or not tk:
            continue
        if skill_to_section is not None:
            matched = skill_to_section.get(tk)
            if matched is None:      # 执行提醒必须绑定现存 item
                continue
            ts = matched
        key = _norm_key(f"{ts}\n{tk}")
        item = {"target_section": ts, "target_skill": tk, "note": note}
        if key in idx_by_key:        # 每 (section, skill) 至多一条
            out[idx_by_key[key]] = item
        else:
            idx_by_key[key] = len(out)
            out.append(item)
    return out


def normalize_state(state):
    """把任意 LLM 产出的 state 收敛到合法结构：固定四板块、去重、notes 绑定现存 item。"""
    if not isinstance(state, dict):
        return empty_state()
    norm = empty_state()
    norm["version"] = int(state.get("version", 1) or 1)
    norm["updated_at_utc"] = state.get("updated_at_utc") or _utc_now()

    items_by_title = {t: [] for t in SECTION_TITLES}
    for sec in (state.get("sections", []) or []):
        if not isinstance(sec, dict):
            continue
        canonical = _ALLOWED_TITLE_MAP.get(_title_key(sec.get("title", "")))
        if canonical is None:
            continue
        items_by_title[canonical].extend(
            [str(i).strip() for i in sec.get("items", []) if str(i).strip()])

    norm["sections"] = [{"title": t, "items": _dedupe_preserve_order(items_by_title[t])}
                        for t in SECTION_TITLES]
    s2s = {}
    for sec in norm["sections"]:
        for it in sec["items"]:
            s2s.setdefault(it, sec["title"])
    norm["execution_notes"] = _normalize_notes(state.get("execution_notes", []), s2s)
    # 二维路由 policy 与手册绑定（由探针搜索写入，非 LLM 产出，需在 normalize 中保留）
    if isinstance(state.get("policy"), dict) and state["policy"]:
        norm["policy"] = dict(state["policy"])
    if state.get("policy_default"):
        norm["policy_default"] = state["policy_default"]
    return norm


def cold_start_manual():
    """冷启动：空 manual（对齐 EmbodiSkill 无种子技能）。"""
    return Manual(empty_state())
