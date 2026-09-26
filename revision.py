#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""技能修订（LLM 驱动，忠实移植 EmbodiSkill revise_manual）。

窗口末尾把缓冲的反思落到 manual：
  1) 拆分：S_NEW/S_BETTER/FAIL_SKILL -> 板块修订；FAIL_EXECUTION -> 执行提醒修订。
  2) LLM 板块修订（sections_revision）：合并/去重/消解冲突，产出新 sections。
  3) LLM 执行提醒修订（execution_notes_revision）：把 FAIL_EXECUTION 变成提醒，绑定现存 item。
  4) 超预算时 LLM refactor 精简。
  版本号 +1。skill-unaware 消融：全部当板块反思，无执行提醒。
"""
import copy
import re

from . import config as C
from . import llm as L
from . import prompts as P
from . import manual as MAN


# E4 程序化黑名单：抽象推演类规则（forward vector / 度数旋转 / 坐标系等）
# 已被逐题归因证实对 4B 直答执行器有害，修订/refactor 后强制删除命中规则。
_BLACKLIST = re.compile(
    r"forward[\s-]+vector"
    r"|coordinate\s+(frame|system|axes)"
    r"|rotat\w*\s+(the\s+)?(frame|axes|reference)"
    r"|mental(ly)?\s+rotat"
    r"|\b(90|180|270)\s*(\u00b0|deg(rees?)?)"
    r"|\bC?CW\b|counter-?clockwise|\bclockwise\b"
    r"|\bbearing\b|\bazimuth\b",
    re.IGNORECASE)


def _filter_blacklist(sections, notes):
    """删除命中黑名单的板块规则与执行提醒；悬空 notes 由末尾 normalize_state 清理。"""
    dropped = []
    for sec in sections:
        keep = []
        for it in sec.get("items", []):
            (dropped if _BLACKLIST.search(it) else keep).append(it)
        sec["items"] = keep
    notes_keep = [n for n in notes if not _BLACKLIST.search(
        f"{n.get('note', '')} {n.get('target_skill', '')}")]
    n_drop = len(dropped) + len(notes) - len(notes_keep)
    if n_drop:
        print(f"[revise] 黑名单过滤 {n_drop} 条抽象推演规则/提醒："
              f"{[d[:60] for d in dropped[:3]]}", flush=True)
    return sections, notes_keep


def _enforce_budget(sections, prev_keys, prev_n, protected_keys):
    """净增预算兜底：修订后总条目至多 prev_n + EVOLVE_SNEW_BUDGET。
    超出部分从尾部删除"新增且非受保护"的条目（改写类不增计数，不受影响）。"""
    total = sum(len([i for i in s.get("items", []) if str(i).strip()]) for s in sections)
    overflow = total - (prev_n + C.EVOLVE_SNEW_BUDGET)
    if overflow <= 0:
        return sections
    dropped = 0
    for sec in reversed(sections):
        items = sec.get("items", [])
        keep = []
        for it in reversed(items):
            k = MAN.item_key(it)
            drop_ok = (dropped < overflow and k not in prev_keys
                       and k not in protected_keys)
            if drop_ok:
                dropped += 1
            else:
                keep.append(it)
        sec["items"] = list(reversed(keep))
    if dropped:
        print(f"[revise] 净增预算：丢弃 {dropped} 条超预算新规则", flush=True)
    return sections


def _protected_text(protected):
    if not protected:
        return "(none)"
    lines = [f"[{title}] {text}" for title, text in protected]
    return "\n".join(lines)


def _restore_protected(sections, protected):
    """受保护规则（连续存活多窗、已验证有效）若被 refactor/截断/blacklist 误删，原样回插。"""
    if not protected:
        return sections
    present = {MAN.item_key(it) for sec in sections for it in sec.get("items", [])}
    by_title = {sec.get("title"): sec for sec in sections}
    restored = 0
    for title, text in protected:
        if MAN.item_key(text) in present or title not in by_title:
            continue
        by_title[title].setdefault("items", []).append(text)
        present.add(MAN.item_key(text))
        restored += 1
    if restored:
        print(f"[revise] 回插 {restored} 条被误删的受保护规则", flush=True)
    return sections


def _serialize(reflections):
    blocks = []
    for i, rf in enumerate(reflections, 1):
        blocks.append("\n".join([
            f"- Reflection ID: {i}",
            f"  Type: {rf.get('reflection_type')}",
            f"  Target Section: {rf.get('target_section')}",
            f"  Target Skill: {rf.get('target_skill')}",
            f"  Content: {str(rf.get('content', '')).strip()}",
        ]))
    return "\n\n".join(blocks)


def _split(reflections):
    sec_types = {"S_NEW", "S_BETTER", "FAIL_SKILL"}
    sec = [r for r in reflections if str(r.get("reflection_type", "")).upper() in sec_types]
    notes = [r for r in reflections if str(r.get("reflection_type", "")).upper() == "FAIL_EXECUTION"]
    return sec, notes


def _revise_sections(manual, sec_reflections):
    user = P.sections_revision_user_prompt.format(
        manual_text=manual.sections_text(),
        reflections_text=_serialize(sec_reflections))
    payload, err = L.chat_json(P.sections_revision_system_prompt, user,
                               max_tokens=C.REVISION_MAX_TOKENS)
    if payload is None or "sections" not in payload:
        print(f"[revise] 板块修订失败，保留原册：{err}", flush=True)
        return manual.sections()
    norm = MAN.normalize_state({"version": manual.version, "sections": payload["sections"],
                                "execution_notes": manual.execution_notes()})
    return norm["sections"]


def _revise_notes(sections, cur_notes, note_reflections):
    tmp = MAN.Manual({"version": 1, "sections": sections, "execution_notes": cur_notes})
    user = P.execution_notes_revision_user_prompt.format(
        manual_sections_text=tmp.sections_text(),
        execution_notes_text=tmp.notes_text(),
        reflections_text=_serialize(note_reflections))
    payload, err = L.chat_json(P.execution_notes_revision_system_prompt, user,
                               max_tokens=C.REVISION_MAX_TOKENS)
    if payload is None or "execution_notes" not in payload:
        print(f"[revise] 执行提醒修订失败，保留原提醒：{err}", flush=True)
        return cur_notes
    norm = MAN.normalize_state({"version": 1, "sections": sections,
                                "execution_notes": payload["execution_notes"]})
    return norm["execution_notes"]


def _maybe_refactor(sections, notes, protected=None):
    tmp = MAN.Manual({"version": 1, "sections": sections, "execution_notes": notes})
    if tmp.n_items() > C.MANUAL_SECTIONS_REFACTOR_THRESHOLD:
        print(f"[revise] 板块 {tmp.n_items()} 条超阈值，触发 refactor（受保护 {len(protected or [])} 条）",
              flush=True)
        payload, err = L.chat_json(
            P.sections_refactor_system_prompt,
            P.sections_refactor_user_prompt.format(
                protected_text=_protected_text(protected),
                manual_text=tmp.sections_text()),
            max_tokens=C.REFACTOR_MAX_TOKENS)
        if payload and "sections" in payload:
            sections = MAN.normalize_state({"version": 1, "sections": payload["sections"],
                                            "execution_notes": notes})["sections"]
            tmp = MAN.Manual({"version": 1, "sections": sections, "execution_notes": notes})
        # 程序化硬性截断（保险丝）：受保护规则优先保留，再按位置裁剪非保护条目
        cap_per_section = 10  # 4 板块 × 10 = 40 条熔断线
        if tmp.n_items() > cap_per_section * len(C.MANUAL_SECTION_TITLES):
            pkeys = {MAN.item_key(t) for _, t in (protected or [])}
            for sec in sections:
                items = sec.get("items", [])
                if len(items) > cap_per_section:
                    prot = [i for i in items if MAN.item_key(i) in pkeys]
                    rest = [i for i in items if MAN.item_key(i) not in pkeys]
                    sec["items"] = (prot + rest)[:max(cap_per_section, len(prot))]
            tmp = MAN.Manual({"version": 1, "sections": sections, "execution_notes": notes})
            print(f"[revise] LLM 压缩不足，程序截断至 {tmp.n_items()} 条", flush=True)
    if tmp.n_notes() > C.EXECUTION_NOTES_REFACTOR_THRESHOLD:
        print(f"[revise] 执行提醒 {tmp.n_notes()} 条超阈值，触发 refactor", flush=True)
        payload, err = L.chat_json(
            P.execution_notes_refactor_system_prompt,
            P.execution_notes_refactor_user_prompt.format(
                manual_sections_text=tmp.sections_text(),
                execution_notes_text=tmp.notes_text()),
            max_tokens=C.REFACTOR_MAX_TOKENS)
        if payload and "execution_notes" in payload:
            notes = MAN.normalize_state({"version": 1, "sections": sections,
                                         "execution_notes": payload["execution_notes"]})["execution_notes"]
    return sections, notes


def revise_manual(manual, reflections, aware=True, protected=None):
    """把一批反思落到 manual，返回新的 Manual（版本 +1）。无反思则原样返回。

    protected: [(section_title, item_text), ...] 连续存活多窗的受保护规则，
    refactor/截断不得改动；证据驱动的定向修订（FAIL_SKILL/S_BETTER 指名）仍允许。"""
    reflections = [r for r in reflections if r and str(r.get("content", "")).strip()]
    if not reflections:
        return manual
    protected = list(protected or [])
    prev_keys = {MAN.item_key(it) for sec in manual.sections()
                 for it in sec.get("items", [])}
    prev_n = manual.n_items()
    protected_keys = {MAN.item_key(t) for _, t in protected}

    sec_reflections, note_reflections = _split(reflections)
    if not aware:
        # 消融：不区分 FAIL_EXECUTION，全部当板块反思，无执行提醒
        sec_reflections = reflections
        note_reflections = []

    sections = copy.deepcopy(manual.sections())
    notes = copy.deepcopy(manual.execution_notes())

    if sec_reflections:
        sections = _revise_sections(manual, sec_reflections)
        # 净增预算兜底：防 S_NEW 洪泛稀释手册（prompt 约束的程序化保险）
        sections = _enforce_budget(sections, prev_keys, prev_n, protected_keys)
    if aware and note_reflections:
        notes = _revise_notes(sections, notes, note_reflections)
    if not aware:
        notes = []

    # 受保护名单收敛到"修订后仍存活"的规则：证据驱动的定向删除不回插，
    # 只拦 refactor/截断/blacklist 这些盲目环节的误删
    live_keys = {MAN.item_key(it) for sec in sections for it in sec.get("items", [])}
    protected = [(t, txt) for t, txt in protected if MAN.item_key(txt) in live_keys]

    # 黑名单先行：坏规则先被点名清除，后面的位置截断几乎不会触发
    sections, notes = _filter_blacklist(sections, notes)
    sections, notes = _maybe_refactor(sections, notes, protected=protected)
    # refactor 可能合并改写出新措辞，再过一道
    sections, notes = _filter_blacklist(sections, notes)
    # 受保护规则最终兜底：任何环节误删都原样回插（防“翻修致死”绞碎已验证规则）
    sections = _restore_protected(sections, protected)

    new_state = MAN.normalize_state({
        "version": manual.version + 1,
        "sections": sections,
        "execution_notes": notes if aware else [],
    })
    return MAN.Manual(new_state)
