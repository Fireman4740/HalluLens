# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Prompt Lego Blocks - Modular components for hybrid prompt generation.

This module contains all the building blocks (enums, templates, constraints)
that can be assembled to create prompts for the creativity vs hallucination benchmark.

Architecture:
    A. CREATIVITY_STYLE - Tone/style for FACTUAL/HYBRID/VERY_CREATIVE
    B. TASK_CONSTRAINTS - Measurable format requirements per task type
    C. LENGTH_BLOCK - Target word count constraint
    D. BANNED_PHRASES - Terms to exclude (benchmark contamination)
    E. BLUEPRINT_SCHEMA - JSON schema for blueprint generation
    F. FINAL_PROMPT_SCHEMA - JSON schema for naturalized output
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any


# ==============================================================================
# 1) EXPERIMENT ENUMS
# ==============================================================================


class CreativityLevel(str, Enum):
    """
    Creativity levels for prompt generation.

    Controls the stylistic tone of generated prompts:
    - FACTUAL: Neutral, informative, straightforward
    - HYBRID: Clear with light creative flourishes
    - VERY_CREATIVE: Highly stylized, distinctive voice
    """

    FACTUAL = "FACTUAL"
    HYBRID = "HYBRID"
    VERY_CREATIVE = "VERY_CREATIVE"


class TaskName(str, Enum):
    """
    Task types with measurable format constraints.

    Each task has specific structural requirements (number of sections,
    items, markers) that can be objectively verified in the output.
    """

    INTERVIEW = "INTERVIEW"
    NEWS_ARTICLE = "NEWS_ARTICLE"
    MUSEUM_AUDIO_GUIDE = "MUSEUM_AUDIO_GUIDE"
    PODCAST_SCRIPT = "PODCAST_SCRIPT"
    MEETING_MINUTES = "MEETING_MINUTES"
    FAQ_HELP_PAGE = "FAQ_HELP_PAGE"
    STUDY_GUIDE = "STUDY_GUIDE"
    LESSON_PLAN = "LESSON_PLAN"
    DOCUMENTARY_VOICEOVER = "DOCUMENTARY_VOICEOVER"
    PRESS_RELEASE = "PRESS_RELEASE"
    INTERNAL_BRIEFING_MEMO = "INTERNAL_BRIEFING_MEMO"
    SOCIAL_MEDIA_THREAD = "SOCIAL_MEDIA_THREAD"
    DEBATE_PREP_BRIEF = "DEBATE_PREP_BRIEF"
    TOUR_BROCHURE = "TOUR_BROCHURE"
    ENCYCLOPEDIA_ENTRY = "ENCYCLOPEDIA_ENTRY"
    TIMELINE_PANEL = "TIMELINE_PANEL"


# ==============================================================================
# 2) DATA CLASSES (Specs & Configs)
# ==============================================================================


@dataclass(frozen=True)
class PromptSpec:
    """Specification for a single prompt generation."""

    subject: str
    task: TaskName
    creativity_level: CreativityLevel
    length_words: int = 500

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "task": self.task.value,
            "creativity_level": self.creativity_level.value,
            "length_words": self.length_words,
        }


@dataclass
class GenerationConfig:
    """Configuration for the prompt generation pipeline."""

    prompt_writer_model: str
    language: str = "en"
    min_ref_tokens: int = 350
    max_ref_tokens: int = 650
    blueprint_temperature: float = 0.7
    naturalize_temperature: float = 0.7
    # If True, skip LLM naturalization and render a deterministic prompt template
    # that only varies by task/subject/creativity_level/length_words.
    static_user_prompt: bool = False
    top_p: float = 0.9
    max_retries: int = 3
    max_workers: int = 50
    cache_path: Optional[str] = None
    # LM Studio support
    lm_studio_url: Optional[str] = None
    lm_studio_model: Optional[str] = None


@dataclass
class GeneratedPrompt:
    """Output of the prompt generation pipeline."""

    id: str
    user_prompt: str
    reference_excerpt: str
    metadata: Dict[str, Any]
    quality_checks: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_prompt": self.user_prompt,
            "reference_excerpt": self.reference_excerpt,
            "metadata": self.metadata,
            "quality_checks": self.quality_checks,
        }


@dataclass
class AblationConfig:
    """Configuration for ablation studies."""

    name: str
    include_creativity_style: bool = True
    include_task_constraints: bool = True
    include_length_block: bool = True
    include_banned_phrases: bool = True
    creativity_levels: List[CreativityLevel] = field(
        default_factory=lambda: list(CreativityLevel)
    )
    tasks: List[TaskName] = field(default_factory=lambda: list(TaskName))


# ==============================================================================
# 3) LEGO BLOCK A: BANNED PHRASES (Hard Bans)
# ==============================================================================

BANNED_PHRASES: List[str] = [
    "wikipedia",
    "wikidata",
    "benchmark",
    "evaluation",
    "oracle",
    "rag",
    "retrieval",
    "closed-book",
    "hallucination",
    "ground truth",
    "cite sources",
    "citations required",
    "as an ai",
    "system prompt",
    "language model",
    "trained on",
    "knowledge cutoff",
    "i don't have access",
    "i cannot browse",
]


# ==============================================================================
# 4) LEGO BLOCK B: CREATIVITY STYLE (Tone Guide)
# ==============================================================================

CREATIVITY_STYLE: Dict[CreativityLevel, str] = {
    CreativityLevel.FACTUAL: (
        "Tone: neutral, informative, straightforward. "
        "Avoid metaphors, theatrics, and flowery language. "
        "Focus on clarity and precision."
    ),
    CreativityLevel.HYBRID: (
        "Tone: clear and engaging with light creative flourishes "
        "(rhythm, mild imagery, occasional vivid word choice), "
        "staying focused on the topic. Balance accessibility with style."
    ),
    CreativityLevel.VERY_CREATIVE: (
        "Tone: highly stylized and distinctive (strong voice, playful constraints, "
        "vivid imagery, unexpected angles), but still centered on the topic. "
        "The form itself becomes part of the content."
    ),
}


# ==============================================================================
# 5) LEGO BLOCK C: TASK CONSTRAINTS (Measurable Format Requirements)
# ==============================================================================

TASK_CONSTRAINTS: Dict[TaskName, str] = {
    TaskName.INTERVIEW: (
        "Use-case: drafting a Q&A for a community newsletter; accessible, skimmable.\n"
        "Format requirements:\n"
        "- Exactly 8 Q/A pairs.\n"
        "- Use markers: Q1:, A1:, ... Q8:, A8:.\n"
        "- Keep answers concise but informative (2-4 sentences each).\n"
        "- Questions should progress from basic to more nuanced.\n"
    ),
    TaskName.NEWS_ARTICLE: (
        "Use-case: news-style explainer for a general audience on deadline.\n"
        "Format requirements:\n"
        "- Headline (one line, compelling).\n"
        "- Deck: exactly 2 sentences summarizing the key points.\n"
        "- 3 sections with clear subheadings.\n"
        "- Include exactly 2 short direct quotes in double quotes (attributed to named sources).\n"
        "- End with 1 forward-looking perspective sentence.\n"
    ),

    TaskName.FAQ_HELP_PAGE: (
        "Use-case: help/FAQ page for a public-facing educational site.\n"
        "Format requirements:\n"
        "- Title at the top.\n"
        "- Quick Summary: exactly 3 bullet points.\n"
        "- FAQ: exactly 8 Q/A entries labeled Q1-Q8.\n"
        "- Common Misconceptions section: exactly 4 bullet points.\n"
    ),
    TaskName.STUDY_GUIDE: (
        "Use-case: exam prep material, structured and test-ready.\n"
        "Format requirements:\n"
        "- Learning Objectives: exactly 5 bullets.\n"
        "- Key Terms: exactly 10 items (format: 'Term - one-line definition').\n"
        "- Summary: 1 paragraph (150-200 words).\n"
        "- Mini-quiz: exactly 6 questions (4 multiple-choice + 2 short-answer).\n"
        "- Answer key at the end.\n"
    ),
    TaskName.LESSON_PLAN: (
        "Use-case: teacher lesson plan for a class.\n"
        "Format requirements:\n"
        "- Grade level specified at top.\n"
        "- Learning objectives: exactly 3.\n"
        "- Materials needed: list format.\n"
        "- 1 activity with time allocations.\n"
        "- Assessment criteria section.\n"
    ),
    TaskName.PRESS_RELEASE: (
        "Use-case: official press release announcement.\n"
        "Format requirements:\n"
        "- FOR IMMEDIATE RELEASE header.\n"
        "- Headline and subheadline.\n"
        "- Location and date line (use placeholder: [CITY, DATE]).\n"
        "- Exactly 3 quotes from named sources.\n"
        "- Boilerplate 'About' section at end.\n"
        "- Contact information placeholder.\n"
    ),

    TaskName.SOCIAL_MEDIA_THREAD: (
        "Use-case: social media educational thread, punchy and mobile-friendly.\n"
        "Format requirements:\n"
        "- Exactly 10 posts labeled 1/10 ... 10/10.\n"
        "- Each post max 240 characters.\n"
        "- Include exactly 2 hashtags total across the whole thread.\n"
        "- No links or URLs.\n"
        "- Hook in first post, payoff in last.\n"
    ),
    TaskName.DEBATE_PREP_BRIEF: (
        "Use-case: debate preparation briefing document.\n"
        "Format requirements:\n"
        "- Topic statement at top.\n"
        "- PRO arguments: exactly 4 numbered points with evidence.\n"
        "- CON arguments: exactly 4 numbered points with evidence.\n"
        "- Key statistics: exactly 3 (can be approximate/illustrative).\n"
        "- Anticipated rebuttals: exactly 3.\n"
    ),
}


# ==============================================================================
# 6) LEGO BLOCK D: LENGTH CONSTRAINT
# ==============================================================================

LENGTH_BLOCK: str = "Target length: {length_words} words (plus or minus 15%)."


# ==============================================================================
# 7) LEGO BLOCK E: BLUEPRINT SCHEMA
# ==============================================================================

BLUEPRINT_SCHEMA_DESC: str = """
Return STRICT JSON (no markdown, no explanation) with this exact structure:
{
  "scenario_hook": "1-2 sentences describing a realistic user situation",
  "audience": "2-6 words describing target audience",
  "intent": "what the user wants to achieve (1 sentence)",
  "constraints_soft": ["3-6 non-measurable hints or preferences"],
  "seed_topics": ["3-6 short topic phrases relevant to the subject"],
  "tone_hints": ["2-3 adjectives for desired tone"]
}
""".strip()


BLUEPRINT_PROMPT: str = """You are a professional prompt designer creating realistic USER requests.

Topic: "{subject}"

You are given a reference excerpt ONLY to help you understand the subject and invent plausible angles.

CRITICAL RULES:
- Do NOT quote or copy any sentence from the reference.
- Do NOT mention Wikipedia, sources, benchmarking, evaluation, or retrieval.
- Avoid precise dates unless absolutely essential; prefer durable context.
- The final prompt will be used without any provided documents.
- Keep the scenario realistic - something a real person would ask.

{schema}

Reference excerpt (for context only, DO NOT copy):
---
{reference}
---

Return ONLY valid JSON, no other text.
"""


# ==============================================================================
# 8) LEGO BLOCK F: FINAL PROMPT SCHEMA
# ==============================================================================

FINAL_PROMPT_SCHEMA_DESC: str = """
Return STRICT JSON (no markdown, no explanation) with this exact structure:
{
  "user_prompt": "a single natural user message in English",
  "quality_checks": {
    "mentions_wikipedia": false,
    "has_measurable_constraints": true,
    "includes_subject": true,
    "natural_sounding": true
  }
}
""".strip()


NATURALIZE_PROMPT: str = """You generate standardized user prompts for a benchmark.

The prompt MUST keep the same overall form across examples.
Only these parameters should vary:
- SUBJECT: "{subject}"
- TASK: {task_name}
- CREATIVITY_LEVEL: {creativity_level}
- LENGTH_WORDS: {length_words}

Write ONE user message in English using the exact structure below.
Do NOT add extra backstory, names, dates, locations, or personal context.

USER MESSAGE TEMPLATE (keep section headers and ordering identical):
Task: {task_name}
Subject: {subject}
CreativityLevel: {creativity_level}

Requirements:
{task_constraints}

Length:
{length_block}

Style:
{creativity_style}

Please follow the requirements exactly.

{schema}

Return ONLY valid JSON, no other text.
"""


STATIC_USER_PROMPT_TEMPLATE: str = """I want you to produce a {task_name} about: "{subject}".

--- CONTEXT & REQUIREMENTS (follow exactly) ---
{task_constraints}

--- LENGTH (important) ---
{length_block}

--- TONE / STYLE GUIDE ---
{creativity_style}

--- OUTPUT RULES (strict) ---
- Write the deliverable in English.
- Output ONLY the final content (no preface, no commentary, no explanations).
- Follow all required structure and counts exactly (sections, labels, number of items, timestamps, etc.).
- If you're unsure about a specific detail, keep it high-level or phrase it cautiously rather than guessing.

(Do a quick silent self-check before sending: structure ✓ counts ✓ length ✓)


"""


# ==============================================================================
# 9) HELPER FUNCTIONS
# ==============================================================================


def get_all_tasks() -> List[TaskName]:
    """Return all available task types."""
    return list(TaskName)


def get_all_creativity_levels() -> List[CreativityLevel]:
    """Return all creativity levels."""
    return list(CreativityLevel)


def get_task_constraint(task: TaskName) -> str:
    """Get the constraint string for a specific task."""
    return TASK_CONSTRAINTS.get(task, "")


def get_creativity_style(level: CreativityLevel) -> str:
    """Get the style guide for a creativity level."""
    return CREATIVITY_STYLE.get(level, "")


def check_banned_phrases(text: str) -> List[str]:
    """Check if text contains any banned phrases. Returns list of found phrases."""
    text_lower = text.lower()
    return [phrase for phrase in BANNED_PHRASES if phrase in text_lower]


def validate_task_name(name: str) -> Optional[TaskName]:
    """Validate and return TaskName from string, or None if invalid."""
    try:
        return TaskName(name.upper())
    except ValueError:
        return None


def validate_creativity_level(level: str) -> Optional[CreativityLevel]:
    """Validate and return CreativityLevel from string, or None if invalid."""
    try:
        return CreativityLevel(level.upper())
    except ValueError:
        return None


def render_final_user_message(spec: PromptSpec) -> str:
    """
    Render a deterministic final user prompt.

    Used when GenerationConfig.static_user_prompt=True so the output prompt keeps the
    same shape and only varies by Task/subject/CreativityLevel/length_words.
    """
    return STATIC_USER_PROMPT_TEMPLATE.format(
        task_name=spec.task.value,
        subject=spec.subject,
        creativity_level=spec.creativity_level.value,
        task_constraints=TASK_CONSTRAINTS[spec.task].strip(),
        length_block=LENGTH_BLOCK.format(length_words=spec.length_words),
        creativity_style=CREATIVITY_STYLE[spec.creativity_level].strip(),
    )
