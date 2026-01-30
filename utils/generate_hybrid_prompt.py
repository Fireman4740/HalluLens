# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Hybrid Prompt Generator - Generate prompts for creativity vs hallucination benchmark.

Pipeline:
    DocDB (Wikipedia) ──> pick_reference_excerpt (INTERNAL)
                               │
                               ▼
                    (1) BLUEPRINT JSON (prompt-writer LLM)
                               │
                               ▼
      (2) NATURALIZE → USER PROMPT FINAL (prompt-writer LLM)
                               │
                               ▼
   Dataset: user_prompt (closed-book) + reference (oracle) + meta

Output format matches the existing longwiki pipeline (generation.jsonl):
{
    "prompt": "...",           # The user prompt for the model to answer
    "title": "...",            # Wikipedia article title (subject)
    "reference": "...",        # Reference excerpt for evaluation
    "h_score": ...,            # Wikipedia popularity score
    "h_score_cat": ...,        # H-score category (0-9)
    "pageid": ...,             # Wikipedia page ID
    "revid": ...,              # Wikipedia revision ID
    "task": "...",             # Task type (INTERVIEW, NEWS_ARTICLE, etc.)
    "creativity_level": "...", # FACTUAL, HYBRID, or VERY_CREATIVE
    "length_words": ...,       # Target word count
    "blueprint": {...},        # Blueprint used for generation (for debugging)
}
"""

from __future__ import annotations

import json
import os
import random
import re
import uuid
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

from utils import lm
from utils.qa_utils import split_doc
from utils.cache import Cache
from utils.prompt_lego_blocks import (
    CreativityLevel,
    TaskName,
    PromptSpec,
    GenerationConfig,
    GeneratedPrompt,
    BANNED_PHRASES,
    CREATIVITY_STYLE,
    TASK_CONSTRAINTS,
    LENGTH_BLOCK,
    BLUEPRINT_SCHEMA_DESC,
    BLUEPRINT_PROMPT,
    FINAL_PROMPT_SCHEMA_DESC,
    NATURALIZE_PROMPT,
    render_final_user_message,
    check_banned_phrases,
)


# ==============================================================================
# PROMPT ASSEMBLERS
# ==============================================================================


def make_blueprint_prompt(subject: str, reference: str) -> str:
    """Assemble the blueprint generation prompt."""
    return BLUEPRINT_PROMPT.format(
        subject=subject,
        reference=reference.strip(),
        schema=BLUEPRINT_SCHEMA_DESC,
    )


def make_naturalize_prompt(spec: PromptSpec, blueprint: dict) -> str:
    """Assemble the naturalization prompt with all lego blocks."""
    return NATURALIZE_PROMPT.format(
        subject=spec.subject,
        task_name=spec.task.value,
        creativity_level=spec.creativity_level.value,
        length_words=spec.length_words,
        task_constraints=TASK_CONSTRAINTS[spec.task].strip(),
        length_block=LENGTH_BLOCK.format(length_words=spec.length_words),
        creativity_style=CREATIVITY_STYLE[spec.creativity_level].strip(),
        schema=FINAL_PROMPT_SCHEMA_DESC,
    )


# ==============================================================================
# JSON PARSING UTILITIES
# ==============================================================================


def parse_json_response(response: str) -> Optional[dict]:
    """
    Parse JSON from LLM response with multiple fallback strategies.

    Handles:
    - Clean JSON
    - JSON in markdown code blocks
    - JSON embedded in other text
    """
    if not response:
        return None

    response = response.strip()

    # Strategy 1: Direct parse
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract from markdown code blocks
    code_block_patterns = [
        r"```json\s*\n?(.*?)\n?```",
        r"```\s*\n?(.*?)\n?```",
    ]
    for pattern in code_block_patterns:
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                continue

    # Strategy 3: Find JSON object in response
    # Look for outermost braces
    brace_start = response.find("{")
    brace_end = response.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        json_candidate = response[brace_start : brace_end + 1]
        try:
            return json.loads(json_candidate)
        except json.JSONDecodeError:
            pass

    # Strategy 4: Try to fix common issues
    # Remove trailing commas before closing braces
    cleaned = re.sub(r",\s*}", "}", response)
    cleaned = re.sub(r",\s*]", "]", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    return None


def validate_blueprint(blueprint: dict) -> bool:
    """Validate blueprint structure has required fields."""
    required_keys = {"scenario_hook", "audience", "intent"}
    if not all(key in blueprint for key in required_keys):
        return False

    # Check that required fields are non-empty strings
    for key in required_keys:
        if not isinstance(blueprint.get(key), str) or not blueprint[key].strip():
            return False

    return True


def validate_final_prompt(result: dict, spec: PromptSpec) -> Tuple[bool, List[str]]:
    """
    Validate final prompt output.

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    if "user_prompt" not in result:
        issues.append("Missing 'user_prompt' field")
        return False, issues

    prompt = result["user_prompt"]
    if not isinstance(prompt, str) or not prompt.strip():
        issues.append("Empty or invalid 'user_prompt'")
        return False, issues

    prompt_lower = prompt.lower()

    # Check for banned phrases
    found_banned = check_banned_phrases(prompt)
    if found_banned:
        issues.append(f"Contains banned phrases: {found_banned}")

    # Check subject is mentioned (fuzzy match)
    subject_words = [w for w in spec.subject.lower().split() if len(w) > 3]
    subject_found = any(word in prompt_lower for word in subject_words)
    if not subject_found and spec.subject.lower() not in prompt_lower:
        issues.append(f"Subject '{spec.subject}' not found in prompt")

    # Check minimum length (at least 50 chars for a reasonable prompt)
    if len(prompt) < 50:
        issues.append("Prompt too short (< 50 chars)")

    return len(issues) == 0, issues


def check_reference_leakage(
    prompt: str, reference: str, threshold: float = 0.7
) -> bool:
    """
    Check if the prompt contains copied text from the reference.

    Uses simple substring matching to detect leakage.
    Returns True if leakage is detected.
    """
    # Normalize texts
    prompt_lower = prompt.lower()
    reference_lower = reference.lower()

    # Check for long common substrings (> 30 chars)
    min_leak_length = 30
    for i in range(len(reference_lower) - min_leak_length):
        substring = reference_lower[i : i + min_leak_length]
        if substring in prompt_lower:
            return True

    return False


# ==============================================================================
# HYBRID PROMPT GENERATOR CLASS
# ==============================================================================


class HybridPromptGenerator:
    """
    Main orchestrator for hybrid prompt generation.

    Generates prompts in two phases:
    1. Blueprint: Create structured scenario from subject + reference
    2. Naturalize: Convert blueprint to natural user prompt

    Output format is compatible with the existing longwiki pipeline.
    """

    def __init__(self, config: GenerationConfig):
        self.config = config
        self.model = config.prompt_writer_model

        # Initialize tokenizer for text splitting
        if AutoTokenizer is not None:
            try:
                self.encoding = AutoTokenizer.from_pretrained(
                    config.prompt_writer_model, trust_remote_code=True
                )
            except Exception:
                try:
                    self.encoding = AutoTokenizer.from_pretrained("gpt2")
                except Exception:
                    self.encoding = None
        else:
            self.encoding = None

        # Fallback tokenizer if transformers not available
        if self.encoding is None:
            print("Warning: No tokenizer available, using word-based splitting")

        # Initialize cache if path provided
        self.cache = None
        if config.cache_path:
            try:
                cache_dir = Path(config.cache_path).parent
                cache_dir.mkdir(parents=True, exist_ok=True)
                self.cache = Cache(config.cache_path)
            except Exception as e:
                print(f"Warning: Could not initialize cache: {e}")

    def _generate_llm(
        self, prompt: str, temperature: float, max_tokens: int = 1024
    ) -> str:
        """Generate text using the configured LLM."""
        return lm.generate(
            prompt,
            self.model,
            temperature=temperature,
            top_p=self.config.top_p,
            max_tokens=max_tokens,
        )

    def pick_reference_excerpt(
        self, document: str, language: str = "en"
    ) -> Optional[str]:
        """Select a reference excerpt from the document."""
        if self.encoding is None:
            # Fallback: split by paragraphs and word count
            paragraphs = [p.strip() for p in document.split("\n\n") if p.strip()]
            valid_sections = []
            for para in paragraphs:
                word_count = len(para.split())
                if (
                    self.config.min_ref_tokens
                    <= word_count
                    <= self.config.max_ref_tokens
                ):
                    valid_sections.append(para)

            if not valid_sections:
                # Try combining short paragraphs
                combined = ""
                for para in paragraphs:
                    combined += para + " "
                    word_count = len(combined.split())
                    if word_count >= self.config.min_ref_tokens:
                        if word_count <= self.config.max_ref_tokens:
                            valid_sections.append(combined.strip())
                        break

            if valid_sections:
                return random.choice(valid_sections)
            return document[:2000] if document else None

        # Use proper tokenizer-based splitting
        sections = split_doc(
            document,
            language,
            self.encoding,
            keep_end=False,
            keep_colon=False,
            MIN_LEN=self.config.min_ref_tokens,
            MAX_LEN=self.config.max_ref_tokens,
        )

        if not sections:
            return None

        # Remove last section (often references/see also)
        if len(sections) > 2:
            sections = sections[:-1]

        return random.choice(sections)

    def generate_blueprint(self, subject: str, reference: str) -> Optional[dict]:
        """Generate blueprint JSON from subject and reference."""
        prompt = make_blueprint_prompt(subject, reference)

        for attempt in range(self.config.max_retries):
            try:
                response = self._generate_llm(
                    prompt,
                    temperature=self.config.blueprint_temperature,
                    max_tokens=512,
                )

                blueprint = parse_json_response(response)
                if blueprint and validate_blueprint(blueprint):
                    return blueprint

                if attempt < self.config.max_retries - 1:
                    print(f"  Blueprint attempt {attempt + 1} failed, retrying...")

            except Exception as e:
                print(f"Blueprint generation error (attempt {attempt + 1}): {e}")

        return None

    def naturalize_prompt(
        self, spec: PromptSpec, blueprint: dict, reference: str
    ) -> Optional[dict]:
        """Naturalize blueprint into final user prompt."""
        prompt = make_naturalize_prompt(spec, blueprint)

        for attempt in range(self.config.max_retries):
            try:
                response = self._generate_llm(
                    prompt,
                    temperature=self.config.naturalize_temperature,
                    max_tokens=1024,
                )

                result = parse_json_response(response)
                if result:
                    is_valid, issues = validate_final_prompt(result, spec)

                    # Check for reference leakage
                    if "user_prompt" in result:
                        if check_reference_leakage(result["user_prompt"], reference):
                            issues.append("Reference leakage detected")
                            is_valid = False

                    if is_valid:
                        return result

                    if attempt < self.config.max_retries - 1:
                        print(f"  Naturalize attempt {attempt + 1} failed: {issues}")

            except Exception as e:
                print(f"Naturalize error (attempt {attempt + 1}): {e}")

        return None

    def generate_single(
        self, spec: PromptSpec, reference: str, wiki_metadata: Optional[dict] = None
    ) -> Optional[GeneratedPrompt]:
        """Generate a single hybrid prompt."""
        # Check cache
        cache_key = (
            f"{spec.subject}:{spec.task.value}:{spec.creativity_level.value}:"
            f"{spec.length_words}:static={self.config.static_user_prompt}"
        )
        if self.cache:
            cached = self.cache.get_item(cache_key)
            if cached:
                return GeneratedPrompt(**cached)

        # Deterministic mode: skip LLM blueprint + naturalize and just render a fixed template.
        if self.config.static_user_prompt:
            user_prompt = render_final_user_message(spec)
            output = GeneratedPrompt(
                id=str(uuid.uuid4()),
                user_prompt=user_prompt,
                reference_excerpt=reference,
                metadata={
                    "subject": spec.subject,
                    "task": spec.task.value,
                    "creativity_level": spec.creativity_level.value,
                    "length_words": spec.length_words,
                    "blueprint": {},
                    **(wiki_metadata or {}),
                },
                quality_checks={
                    "mentions_wikipedia": False,
                    "has_measurable_constraints": True,
                    "includes_subject": True,
                    "natural_sounding": True,
                },
            )

            if self.cache:
                self.cache.set_item(cache_key, output.to_dict())
            return output

        # Generate blueprint
        blueprint = self.generate_blueprint(spec.subject, reference)
        if not blueprint:
            print(f"  Failed to generate blueprint for '{spec.subject}'")
            return None

        # Naturalize to final prompt
        result = self.naturalize_prompt(spec, blueprint, reference)
        if not result:
            print(f"  Failed to naturalize prompt for '{spec.subject}'")
            return None

        # Build output
        output = GeneratedPrompt(
            id=str(uuid.uuid4()),
            user_prompt=result["user_prompt"],
            reference_excerpt=reference,
            metadata={
                "subject": spec.subject,
                "task": spec.task.value,
                "creativity_level": spec.creativity_level.value,
                "length_words": spec.length_words,
                "blueprint": blueprint,
                **(wiki_metadata or {}),
            },
            quality_checks=result.get("quality_checks", {}),
        )

        # Cache result
        if self.cache:
            self.cache.set_item(cache_key, output.to_dict())

        return output

    def generate_single_for_longwiki(
        self,
        wiki_row: dict,
        task: TaskName,
        creativity_level: CreativityLevel,
        length_words: int = 500,
    ) -> Optional[dict]:
        """
        Generate a single prompt in longwiki-compatible format.

        Returns dict matching the existing longwiki jsonl format:
        {
            "prompt": "...",
            "title": "...",
            "reference": "...",
            ...
        }
        """
        # Pick reference excerpt
        reference = self.pick_reference_excerpt(
            wiki_row.get("document", ""), self.config.language
        )
        if not reference:
            return None

        # Create spec
        spec = PromptSpec(
            subject=wiki_row["title"],
            task=task,
            creativity_level=creativity_level,
            length_words=length_words,
        )

        # Wiki metadata
        wiki_metadata = {
            "h_score": wiki_row.get("h_score"),
            "h_score_cat": wiki_row.get("h_score_cat"),
            "pageid": wiki_row.get("pageid"),
            "revid": wiki_row.get("revid"),
            "description": wiki_row.get("description", ""),
            "categories": wiki_row.get("categories", []),
        }

        # Generate
        result = self.generate_single(spec, reference, wiki_metadata)
        if not result:
            return None

        # Convert to longwiki format
        return {
            "prompt": result.user_prompt,
            "title": spec.subject,
            "reference": reference,
            "h_score": wiki_metadata.get("h_score"),
            "h_score_cat": wiki_metadata.get("h_score_cat"),
            "pageid": wiki_metadata.get("pageid"),
            "revid": wiki_metadata.get("revid"),
            "description": wiki_metadata.get("description", ""),
            "categories": wiki_metadata.get("categories", []),
            "task": spec.task.value,
            "creativity_level": spec.creativity_level.value,
            "length_words": spec.length_words,
            "blueprint": result.metadata.get("blueprint", {}),
            "quality_checks": result.quality_checks,
        }

    def generate_batch_longwiki(
        self,
        wiki_data: pd.DataFrame,
        tasks: List[TaskName],
        creativity_levels: List[CreativityLevel],
        length_words: int = 500,
        n_per_combination: Optional[int] = None,
    ) -> List[dict]:
        """
        Generate batch of prompts in longwiki-compatible format.

        Args:
            wiki_data: DataFrame with Wikipedia articles
            tasks: List of task types to generate
            creativity_levels: List of creativity levels
            length_words: Target word count
            n_per_combination: Max samples per task/creativity combo (None = all)

        Returns:
            List of dicts in longwiki format
        """
        results = []

        # Create work items
        work_items = []
        for _, row in wiki_data.iterrows():
            for task in tasks:
                for creativity in creativity_levels:
                    work_items.append(
                        {
                            "row": row.to_dict(),
                            "task": task,
                            "creativity": creativity,
                            "length_words": length_words,
                        }
                    )

        # Limit per combination if specified
        if n_per_combination:
            # Group by task/creativity and limit
            from collections import defaultdict

            grouped = defaultdict(list)
            for item in work_items:
                key = (item["task"].value, item["creativity"].value)
                grouped[key].append(item)

            work_items = []
            for key, items in grouped.items():
                random.shuffle(items)
                work_items.extend(items[:n_per_combination])

        # Generate
        def generate_one(item):
            try:
                return self.generate_single_for_longwiki(
                    wiki_row=item["row"],
                    task=item["task"],
                    creativity_level=item["creativity"],
                    length_words=item["length_words"],
                )
            except Exception as e:
                print(f"Error generating prompt: {e}")
                return None

        if self.config.max_workers > 1:
            results = thread_map(
                generate_one,
                work_items,
                max_workers=self.config.max_workers,
                desc=f"Generating prompts with {self.model}",
            )
        else:
            results = []
            for item in tqdm(work_items, desc="Generating prompts"):
                results.append(generate_one(item))

        # Filter None results
        return [r for r in results if r is not None]


# ==============================================================================
# BATCH GENERATION FUNCTIONS (matching generate_question.py pattern)
# ==============================================================================


def hybrid_prompt_generation_run_batch(
    wiki_input_path: str,
    N: int = 250,
    q_generator: str = "openai/gpt-4o-mini",
    output_path: str = "",
    from_scratch: bool = False,
    low_level: int = 5,
    high_level: int = 10,
    tasks: Optional[List[str]] = None,
    creativity_levels: Optional[List[str]] = None,
    length_words: int = 500,
    static_user_prompt: bool = False,
    lm_studio_url: Optional[str] = None,
    lm_studio_model: Optional[str] = None,
) -> List[dict]:
    """
    Main batch generation function matching the signature of longform_QA_generation_run_batch.

    Args:
        wiki_input_path: Path to wiki data jsonl
        N: Total number of prompts to generate
        q_generator: Model to use for prompt generation
        output_path: Where to save results
        from_scratch: Whether to overwrite existing file
        low_level: Minimum h_score_cat
        high_level: Maximum h_score_cat
        tasks: List of task names (defaults to subset)
        creativity_levels: List of creativity levels (defaults to all)
        length_words: Target word count
        lm_studio_url: Optional LM Studio URL
        lm_studio_model: Optional LM Studio model name

    Returns:
        List of generated prompts in longwiki format
    """
    import jsonlines

    print(f"Starting hybrid prompt generation N={N}...")
    print(f"Wiki Source = {wiki_input_path}")
    print(f"Model = {q_generator}")

    # Setup config
    config = GenerationConfig(
        prompt_writer_model=q_generator,
        max_workers=1,  # Sequential for stability
        static_user_prompt=static_user_prompt,
        cache_path=str(Path(output_path).parent / ".hybrid_cache.db")
        if output_path
        else None,
        lm_studio_url=lm_studio_url,
        lm_studio_model=lm_studio_model,
    )

    generator = HybridPromptGenerator(config)

    # Load existing results if not from_scratch
    existing_results = []
    if output_path and os.path.exists(output_path) and not from_scratch:
        with jsonlines.open(output_path) as reader:
            existing_results = list(reader)
        print(f"Loaded {len(existing_results)} existing prompts")
        if len(existing_results) >= N:
            return existing_results[:N]

    # Load wiki data
    wiki_data_all = pd.read_json(wiki_input_path, orient="records", lines=True)

    # Check for not_exist titles file
    not_exist_path = "data/wiki_data/doc_goodwiki_not_exist_titles.txt"
    if os.path.exists(not_exist_path):
        with open(not_exist_path, "r") as f:
            not_exist = [line.strip() for line in f.readlines()]
        wiki_data_all = wiki_data_all[~wiki_data_all["title"].isin(not_exist)]

    # Parse tasks and creativity levels
    default_tasks = ["INTERVIEW", "NEWS_ARTICLE", "ENCYCLOPEDIA_ENTRY", "FAQ_HELP_PAGE"]
    task_list = [TaskName(t.upper()) for t in (tasks or default_tasks)]

    default_creativity = ["FACTUAL", "HYBRID", "VERY_CREATIVE"]
    creativity_list = [
        CreativityLevel(c.upper()) for c in (creativity_levels or default_creativity)
    ]

    # Calculate per-bin count
    n_bins = high_level - low_level
    per_bin_count = N // n_bins if n_bins > 0 else N

    all_results = list(existing_results)

    # Generate per h_score bin
    for bin_idx in range(low_level, high_level):
        if len(all_results) >= N:
            break

        level_wiki = wiki_data_all[wiki_data_all["h_score_cat"] == bin_idx]
        if level_wiki.empty:
            print(f"Warning: No data for h_score_cat={bin_idx}")
            continue

        level_wiki = level_wiki.sample(frac=1)  # Shuffle

        # Calculate how many we need from this bin
        remaining = N - len(all_results)
        target_for_bin = min(per_bin_count, remaining)

        # Select subset of wiki data
        wiki_subset = level_wiki.head(target_for_bin + 10)  # Buffer

        print(f"\nBin {bin_idx}: generating up to {target_for_bin} prompts...")

        # Generate prompts
        for _, row in tqdm(
            wiki_subset.iterrows(), total=len(wiki_subset), desc=f"Bin {bin_idx}"
        ):
            if len(all_results) >= N:
                break

            # Randomly select task and creativity
            task = random.choice(task_list)
            creativity = random.choice(creativity_list)

            result = generator.generate_single_for_longwiki(
                wiki_row=row.to_dict(),
                task=task,
                creativity_level=creativity,
                length_words=length_words,
            )

            if result:
                all_results.append(result)

                # Save incrementally
                if output_path:
                    output_dir = Path(output_path).parent
                    output_dir.mkdir(parents=True, exist_ok=True)
                    with jsonlines.open(output_path, "w") as writer:
                        for r in all_results:
                            writer.write(r)

        print(f"Bin {bin_idx}: generated {len(all_results)} total prompts")

    print(f"\nFinished. Total prompts: {len(all_results)}")
    return all_results[:N]


# ==============================================================================
# CLI INTERFACE
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate hybrid prompts for creativity vs hallucination benchmark"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/wiki_data/doc_goodwiki_h_score.jsonl",
        help="Path to wiki data jsonl",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/longwiki/save/hybrid_prompts.jsonl",
        help="Output path for generated prompts",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o-mini",
        help="Model for prompt generation",
    )
    parser.add_argument(
        "--N", type=int, default=250, help="Number of prompts to generate"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Task types to generate (e.g., INTERVIEW NEWS_ARTICLE)",
    )
    parser.add_argument(
        "--creativity",
        nargs="+",
        default=None,
        help="Creativity levels (e.g., FACTUAL HYBRID VERY_CREATIVE)",
    )
    parser.add_argument("--length", type=int, default=500, help="Target word count")
    parser.add_argument(
        "--static_user_prompt",
        action="store_true",
        help="Skip LLM naturalization and output a deterministic prompt template",
    )
    parser.add_argument("--low_level", type=int, default=5, help="Minimum h_score_cat")
    parser.add_argument(
        "--high_level", type=int, default=10, help="Maximum h_score_cat"
    )
    parser.add_argument(
        "--from_scratch", action="store_true", help="Overwrite existing output file"
    )
    parser.add_argument(
        "--lm_studio_url",
        type=str,
        default=None,
        help="LM Studio API URL (e.g., http://10.10.12.21:1234/v1/chat/completions)",
    )
    parser.add_argument(
        "--lm_studio_model", type=str, default=None, help="LM Studio model name"
    )

    args = parser.parse_args()

    results = hybrid_prompt_generation_run_batch(
        wiki_input_path=args.input_path,
        N=args.N,
        q_generator=args.model,
        output_path=args.output_path,
        from_scratch=args.from_scratch,
        low_level=args.low_level,
        high_level=args.high_level,
        tasks=args.tasks,
        creativity_levels=args.creativity,
        length_words=args.length,
        static_user_prompt=args.static_user_prompt,
        lm_studio_url=args.lm_studio_url,
        lm_studio_model=args.lm_studio_model,
    )

    print(f"\nGenerated {len(results)} prompts")
    print(f"Saved to: {args.output_path}")
