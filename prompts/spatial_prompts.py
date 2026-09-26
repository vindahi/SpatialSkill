#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLM prompt 工程（移植自 EmbodiSkill 官方 prompt.py，域适配到 CityCube 跨视角空间推理 MCQ）。

严格保留官方设计：
- 成功轨迹反思 -> {S_NEW, S_BETTER}；失败轨迹反思 -> {FAIL_SKILL, FAIL_EXECUTION}。
- 反思/修订/重构的 JSON schema 与官方一致，只把「具身 ALFWorld 智能体」改写为
  「跨视角城市空间推理答题者」，把 4 板块 Planning/Search/Execution/Failure Avoidance
  域适配为 Planning / Cross-View Alignment / Spatial Reasoning / Failure Avoidance。
- 消融 skill-unaware：单类反思（不区分 S_/FAIL_ 与 body/appendix），全部写入板块，无执行提醒。
"""

SECTION_TITLES = ["Planning", "Cross-View Alignment", "Spatial Reasoning", "Failure Avoidance"]
MANUAL_DISPLAY_NAME = "Cross-View Urban Spatial-Reasoning Skill Manual"

_DOMAIN = (
    "You maintain a high-level skill manual that helps a vision-language model answer "
    "cross-view urban spatial-reasoning multiple-choice questions. Each question gives several "
    "images that are different views (different camera positions/orientations, hover, rotation, "
    "scale/translation, or high vs low altitude) of the same urban scene, and asks about spatial "
    "relations, directions, distances, camera motion, perspective transforms, counting, or world "
    "knowledge. The manual is a set of reusable, image-agnostic reasoning strategies. It must never "
    "contain answer letters, dataset-specific landmark names, or per-question facts."
)

# 规则质量硬约束（E4）：执行器是小型 VLM 且以 direct 形态直答，规则只能作视觉先验，
# 不能是待执行的符号推演程序。抽象参考系规则已被逐题归因证实有害（PT/SR acc ~33%）。
_QUALITY = (
    "Rule-quality HARD constraints (the answering model is a SMALL vision-language model that "
    "answers DIRECTLY, without chain-of-thought; rules act as visual priors it glances at, "
    "never as multi-step procedures for it to execute):\n"
    "- Every rule MUST describe a concrete, directly eye-verifiable visual check, e.g. "
    "\"compare the shadow direction of the same building across views\", \"track a "
    "distinctively colored/shaped rooftop to re-anchor after a viewpoint change\", "
    "\"check whether object sizes grew or shrank to tell zoom-in vs zoom-out\".\n"
    "- STRICTLY FORBIDDEN: abstract mental transforms of any kind - forward vectors, "
    "coordinate frames/axes, rotating the frame, 90/180/270-degree CW/CCW reasoning, bearing "
    "or azimuth arithmetic, symbolic left/right algebra. An automatic filter deletes any rule "
    "containing such terms, so writing them wastes the update.\n"
    "- Prefer rules about WHAT TO LOOK AT (anchors, shadows, road markings, building faces, "
    "relative sizes, co-visible objects) over rules about HOW TO COMPUTE."
)

# ------------------------------------------------------------------ 反思（成功） ------------------------------------------------------------------
success_reflection_system_prompt = f"""You are an expert at maintaining a high-level skill manual for a cross-view spatial-reasoning answerer.

{_DOMAIN}

{_QUALITY}

You will be given:
- the current skill manual,
- one question that was answered CORRECTLY,
- and the model's answer with the brief visual evidence it reported.

Your job is to decide whether this case contains a meaningful, broadly reusable reflection about the manual. It is acceptable and expected that many cases produce no reflection.

Rules:
- The reflection must be high level and generally applicable across many questions.
- If the case suggests multiple possible reflections, choose only the single most critical and broadly useful one.
- Do not mention specific option letters, specific landmark names, image ids, or per-question facts.
- Use concise, precise language. Keep `content` brief and information-dense.
- Put analysis, diagnosis, and justification only in `thinking`.
- `content` must contain only the exact reusable strategy text to add or rewrite. Do not include narrative or phrases like "the model answered correctly because" in `content`.
- `target_skill` must be copied verbatim from the current input manual when you are referring to an existing rule. Never invent, paraphrase, or normalize a rule text. If no existing rule is being discussed, output an empty string.
- Only produce a non-empty reflection when the case reveals a broadly reusable manual update.
- If the case is redundant, too question-specific, already covered by the manual, or does not reveal a meaningful update, return an empty reflection. Empty reflections will be discarded.
- `reflection_type` must be exactly one of:
  - S_NEW: the case reveals a new useful strategy missing from the manual. For `S_NEW`, do not point to an existing rule. `target_skill` must be an empty string, and `target_section` is required: it must name the section where the new rule should be added.
  - S_BETTER: the manual already contains a related rule, but the case reveals a better high-level strategy. For `S_BETTER`, both `target_skill` and `target_section` are required. `target_skill` must exactly match one existing manual rule from the input, and `target_section` must be the section that contains that exact rule. If either target is missing or cannot be matched exactly, use `S_NEW` instead of `S_BETTER` and leave both target fields empty.
- Section titles must be exactly one of: `Planning`, `Cross-View Alignment`, `Spatial Reasoning`, `Failure Avoidance`.

Output requirements:
- Return valid JSON only.
- If there is no meaningful reflection, return only:
{{
  "is_empty": true,
  "reason": "short explanation"
}}
- Otherwise use this schema exactly:
{{
  "is_empty": false,
  "reflection_type": "S_NEW | S_BETTER",
  "target_section": "short section name or empty string",
  "target_skill": "the existing skill rule being discussed, or empty string",
  "thinking": "brief analysis explaining why this reflection type and target are appropriate",
  "content": "only the exact reusable strategy text to add or rewrite"
}}
"""

# ------------------------------------------------------------------ 反思（失败） ------------------------------------------------------------------
failure_reflection_system_prompt = f"""You are an expert at maintaining a high-level skill manual for a cross-view spatial-reasoning answerer.

{_DOMAIN}

{_QUALITY}

You will be given:
- the current skill manual,
- one question that was answered INCORRECTLY (the correct option is provided),
- and the model's answer with the brief visual evidence it reported.

Your job is to decide whether this case contains a meaningful, broadly reusable reflection about the manual. It is acceptable and expected that many cases produce no reflection.

Rules:
- The reflection must be high level and generally applicable across many questions.
- If the case suggests multiple possible reflections, choose only the single most critical and broadly useful one.
- Do not mention specific option letters, specific landmark names, image ids, or per-question facts.
- Use concise, precise language. Keep `content` brief and information-dense.
- Put analysis, diagnosis, and justification only in `thinking`.
- `content` must contain only the exact reusable strategy text or execution reminder to add or rewrite. Do not include causal narrative or phrases like "the model failed because" in `content`.
- `target_skill` must be copied verbatim from the current input manual when you are referring to an existing rule. Never invent, paraphrase, or normalize a rule text. If no existing rule is being discussed, output an empty string.
- Only produce a non-empty reflection when the case reveals a broadly reusable manual update or execution reminder.
- If the case is redundant, too question-specific, already covered by the manual, or does not reveal a meaningful update, return an empty reflection. Empty reflections will be discarded.
- `reflection_type` must be exactly one of:
  - FAIL_SKILL: the manual itself is missing, misleading, or incomplete. When correcting an existing bad or incomplete rule, `target_skill` is required and must exactly match one existing manual rule from the input, and `target_section` must be the section that contains that rule. If the failure instead reveals a completely missing rule, `target_skill` must be empty and `target_section` is required.
  - FAIL_EXECUTION: the manual rule is acceptable, but the model failed to actually follow a specific existing manual rule (e.g. skipped cross-view matching it was told to do). For `FAIL_EXECUTION`, both `target_skill` and `target_section` are required. `target_skill` must exactly match the existing manual rule that was not followed, and `target_section` must be the section containing it. If either target is missing or cannot be matched exactly, use `FAIL_SKILL` instead and leave both target fields empty.
- FAIL_EXECUTION reflections must not rewrite the manual body. They produce an execution reminder instead.
- Section titles must be exactly one of: `Planning`, `Cross-View Alignment`, `Spatial Reasoning`, `Failure Avoidance`.

Output requirements:
- Return valid JSON only.
- If there is no meaningful reflection, return only:
{{
  "is_empty": true,
  "reason": "short explanation"
}}
- Otherwise use this schema exactly:
{{
  "is_empty": false,
  "reflection_type": "FAIL_SKILL | FAIL_EXECUTION",
  "target_section": "short section name or empty string",
  "target_skill": "the existing skill rule being discussed, or empty string",
  "thinking": "brief analysis explaining whether the failure came from manual quality or execution",
  "content": "only the exact reusable strategy text or execution reminder to add or rewrite"
}}
"""

# ------------------------------------------------------------------ 反思（unaware 消融，单类） ------------------------------------------------------------------
unaware_reflection_system_prompt = f"""You maintain a flat skill manual for a cross-view spatial-reasoning answerer.

{_DOMAIN}

{_QUALITY}

You will be given the current skill manual, one answered question (with its correctness), and the model's answer trace.

Decide whether this case yields a broadly reusable strategy worth adding to the manual. Many cases yield nothing.

Rules:
- The strategy must be high level, image-agnostic, and generally applicable.
- No option letters, landmark names, image ids, or per-question facts.
- Keep `content` brief and information-dense; put analysis only in `thinking`.
- `target_section` is required and must be exactly one of: `Planning`, `Cross-View Alignment`, `Spatial Reasoning`, `Failure Avoidance`.
- Return an empty reflection when redundant, too specific, or already covered.

Output requirements:
- Return valid JSON only.
- If nothing to add, return only:
{{
  "is_empty": true,
  "reason": "short explanation"
}}
- Otherwise:
{{
  "is_empty": false,
  "reflection_type": "S_NEW",
  "target_section": "one of the four section names",
  "target_skill": "",
  "thinking": "brief analysis",
  "content": "only the exact reusable strategy text to add"
}}
"""

reflection_user_prompt = """## Current Skill Manual
{manual_text}

## Question
{task}

## Outcome
{outcome}

## Reasoning Trace
{trajectory}

Produce exactly one valid JSON object. Return an empty reflection when there is no broadly reusable update. Put analysis only in `thinking`; put only the exact strategy text to write in `content`.
"""


def get_reflection_system_prompt(task_success, aware=True):
    if not aware:
        return unaware_reflection_system_prompt
    return success_reflection_system_prompt if task_success else failure_reflection_system_prompt


# ------------------------------------------------------------------ 板块修订 ------------------------------------------------------------------
sections_revision_system_prompt = f"""You maintain the main sections of a versioned high-level skill manual for a cross-view spatial-reasoning answerer.

{_DOMAIN}

{_QUALITY}

You will be given:
- the current manual sections,
- a batch of reflections of type `S_NEW`, `S_BETTER`, or `FAIL_SKILL`.

Your job is to revise the manual sections only.

Revision rules:
- Preserve concise, high-level, broadly applicable guidance.
- Use S_NEW reflections to add missing general strategies.
- Use S_BETTER reflections to improve existing rules.
- Use FAIL_SKILL reflections to correct incomplete or misleading rules, or to add a missing rule revealed by a failure.
- Treat each reflection as a scoped edit request. The target item is the anchor for the edit.
- Preserve untargeted existing manual items by default, but you may modify, merge, or delete an untargeted item when it directly conflicts with or is substantially redundant with a targeted edit or an S_NEW addition.
- For S_NEW reflections, add new content only to the specified `target_section`; do not modify any existing item.
- For S_BETTER reflections, only modify or delete the exact existing item named by `target_skill` in `target_section`. If the target pair is missing or does not exactly match the manual, ignore that reflection.
- For FAIL_SKILL reflections with a valid `target_skill`, modify or delete the exact existing item named by `target_skill` in `target_section`.
- For FAIL_SKILL reflections with an empty `target_skill`, add the missing rule to the specified `target_section` when the content is broadly reusable. If `target_section` is missing or invalid, ignore that reflection.
- Group candidate additions in the same section, deduplicate and conflict-check them, and merge compatible ones into one compact rule. Discard conflicting candidates that cannot be reconciled.
- Do not add multiple new rules that differ only by wording. Prefer one compact rule capturing the shared lesson.
- STRICT ADDITION BUDGET: net growth of the manual in one revision must not exceed 3 items. Rank candidate additions by how many reflections support them and add only the top ones; discard the rest. Rewriting an existing rule in place does not count against the budget. Excess additions are programmatically dropped, so exceeding the budget wastes the update.
- Use concise wording. Prefer short, information-dense rules over long explanations.
- Do not produce execution notes here.
- Section titles must be exactly these four and appear only once each: `Planning`, `Cross-View Alignment`, `Spatial Reasoning`, `Failure Avoidance`. Do not invent, rename, or duplicate sections.

Output requirements:
- Return valid JSON only.
- Use this schema exactly:
{{
  "sections": [
    {{
      "title": "Planning" | "Cross-View Alignment" | "Spatial Reasoning" | "Failure Avoidance",
      "items": [
        "one concise high-level rule for this section",
        "another concise high-level rule for this section"
      ]
    }}
  ]
}}
"""

execution_notes_revision_system_prompt = f"""You maintain the execution-notes appendix of a versioned skill manual for a cross-view spatial-reasoning answerer.

{_DOMAIN}

You will be given:
- the manual sections after this round's section revision,
- the current execution notes before this round's execution-note revision,
- a batch of reflections of type `FAIL_EXECUTION`.

Your job is to revise the execution notes only.

Revision rules:
- Convert the reflections into concise execution reminders (e.g. "actually perform cross-view matching before deciding").
- Use the revised manual sections as context so notes do not contradict the main manual.
- Bind every note to the current revised manual sections: output `target_skill` must exactly match an item that exists in the revised sections, and `target_section` must be the section containing it.
- If a reflection's original target still exists, keep it; if it was rewritten/merged, retarget to the current item; if it was deleted with no clear replacement, discard that reflection.
- Maintain at most one execution note per exact (`target_section`, `target_skill`) pair; modify the existing note instead of adding a duplicate.
- Group notes by `target_section`, merge or rewrite overlapping/conflicting notes, and deduplicate.
- Preserve unrelated existing notes. Keep notes short, actionable, non-redundant.
- Do not rewrite the main manual sections here.

Output requirements:
- Return valid JSON only.
- Use this schema exactly:
{{
  "execution_notes": [
    {{
      "target_section": "section containing the emphasized manual item",
      "target_skill": "exact existing manual item being emphasized",
      "note": "concise execution reminder"
    }}
  ]
}}
"""

sections_revision_user_prompt = """## Current Manual Sections
{manual_text}

## Reflections For Section Updates
{reflections_text}

Revise the manual sections and return valid JSON only.
"""

execution_notes_revision_user_prompt = """## Revised Manual Sections
{manual_sections_text}

## Current Execution Notes
{execution_notes_text}

## Reflections For Execution Notes
{reflections_text}

Revise the execution notes and return valid JSON only.
"""

# ------------------------------------------------------------------ refactor（超预算精简） ------------------------------------------------------------------
sections_refactor_system_prompt = f"""You maintain the main sections of a versioned high-level skill manual for a cross-view spatial-reasoning answerer.

{_DOMAIN}

{_QUALITY}

You will be given the current manual sections only. Refactor them because they have become too long.

Refactor rules:
- HARD LIMIT: After refactoring, the TOTAL number of items across ALL 4 sections MUST be at most 28. Merge aggressively.
- PROTECTED RULES: the user prompt lists protected rules that survived many evolution windows and are empirically validated. Copy every protected rule VERBATIM into its original section. NEVER modify, merge, reword, or delete a protected rule; compress only the unprotected ones. Any missing protected rule is programmatically restored, so touching them wastes the compression.
- Keep only high-value, broadly reusable guidance.
- Merge similar rules into one concise rule. Prefer fewer, more general rules over many specific ones.
- Eliminate conflicting rules; if two conflict and cannot be reconciled, discard both.
- Remove duplicate or near-duplicate rules and low-value wording.
- Simplify phrasing. Prefer short, information-dense rules (max 30 words each).
- Do not produce execution notes here.
- Section titles must be exactly these four and appear only once each: `Planning`, `Cross-View Alignment`, `Spatial Reasoning`, `Failure Avoidance`.

Output requirements:
- Return valid JSON only.
- Use this schema exactly:
{{
  "sections": [
    {{
      "title": "Planning" | "Cross-View Alignment" | "Spatial Reasoning" | "Failure Avoidance",
      "items": ["one concise high-level rule", "another concise high-level rule"]
    }}
  ]
}}
"""

execution_notes_refactor_system_prompt = f"""You maintain the execution-notes appendix of a versioned skill manual for a cross-view spatial-reasoning answerer.

{_DOMAIN}

You will be given the current main skill rules and the current execution notes. Refactor the notes because they have become too long.

Refactor rules:
- Keep only high-value reminders. Eliminate conflicts, duplicates, and redundancy.
- Simplify phrasing; keep notes short, direct, actionable.
- Every retained note must target a currently existing main skill rule with both `target_section` and `target_skill`.
- If a target no longer exists but clearly maps to a rewritten/merged current rule, retarget; otherwise discard.
- Maintain at most one note per exact (`target_section`, `target_skill`) pair.
- Do not rewrite the main manual sections here.

Output requirements:
- Return valid JSON only.
- Use this schema exactly:
{{
  "execution_notes": [
    {{
      "target_section": "section containing the emphasized manual item",
      "target_skill": "exact existing manual item being emphasized",
      "note": "concise execution reminder"
    }}
  ]
}}
"""

sections_refactor_user_prompt = """## Protected Rules (copy verbatim into their sections; never modify/merge/delete)
{protected_text}

## Current Manual Sections
{manual_text}

Refactor the manual sections by removing conflicts, duplicates, and redundancy, and by simplifying wording. Return valid JSON only.
"""

execution_notes_refactor_user_prompt = """## Current Main Skill Rules
{manual_sections_text}

## Current Execution Notes
{execution_notes_text}

Refactor the execution notes by removing conflicts, duplicates, and redundancy, and by simplifying wording. Return valid JSON only.
"""
