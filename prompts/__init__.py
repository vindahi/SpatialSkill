#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLM prompt 工程包（移植自 EmbodiSkill 官方设计，域适配 CityCube）。"""
from .spatial_prompts import (  # noqa: F401
    SECTION_TITLES,
    MANUAL_DISPLAY_NAME,
    get_reflection_system_prompt,
    reflection_user_prompt,
    sections_revision_system_prompt,
    sections_revision_user_prompt,
    execution_notes_revision_system_prompt,
    execution_notes_revision_user_prompt,
    sections_refactor_system_prompt,
    sections_refactor_user_prompt,
    execution_notes_refactor_system_prompt,
    execution_notes_refactor_user_prompt,
)
