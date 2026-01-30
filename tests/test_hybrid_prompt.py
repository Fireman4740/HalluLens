# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for hybrid prompt generation.

Run with: pytest tests/test_hybrid_prompt.py -v
"""

import json
import pytest
import sys
from pathlib import Path

# Add project root to path
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

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
    check_banned_phrases,
    get_all_tasks,
    get_all_creativity_levels,
    get_task_constraint,
    get_creativity_style,
    validate_task_name,
    validate_creativity_level,
)

from utils.generate_hybrid_prompt import (
    make_blueprint_prompt,
    make_naturalize_prompt,
    parse_json_response,
    validate_blueprint,
    validate_final_prompt,
    check_reference_leakage,
)


# ==============================================================================
# Test Prompt Lego Blocks
# ==============================================================================


class TestCreativityLevel:
    """Tests for CreativityLevel enum."""

    def test_all_levels_defined(self):
        """All three creativity levels should be defined."""
        assert CreativityLevel.FACTUAL.value == "FACTUAL"
        assert CreativityLevel.HYBRID.value == "HYBRID"
        assert CreativityLevel.VERY_CREATIVE.value == "VERY_CREATIVE"

    def test_all_levels_have_style(self):
        """Every creativity level should have a style guide."""
        for level in CreativityLevel:
            assert level in CREATIVITY_STYLE
            assert len(CREATIVITY_STYLE[level]) > 20


class TestTaskName:
    """Tests for TaskName enum."""

    def test_all_tasks_defined(self):
        """All 16 task types should be defined."""
        assert len(TaskName) == 16

    def test_all_tasks_have_constraints(self):
        """Every task should have defined constraints."""
        for task in TaskName:
            assert task in TASK_CONSTRAINTS
            constraint = TASK_CONSTRAINTS[task]
            assert len(constraint) > 50
            # Constraints should contain measurable requirements (numbers)
            assert any(char.isdigit() for char in constraint), (
                f"Task {task} should have measurable constraints with numbers"
            )

    def test_interview_constraints_specific(self):
        """Interview task should have specific constraints."""
        constraint = TASK_CONSTRAINTS[TaskName.INTERVIEW]
        assert "8 Q/A pairs" in constraint or "8" in constraint
        assert "Q1:" in constraint or "markers" in constraint.lower()


class TestPromptSpec:
    """Tests for PromptSpec dataclass."""

    def test_creation(self):
        """Should create PromptSpec correctly."""
        spec = PromptSpec(
            subject="Marie Curie",
            task=TaskName.INTERVIEW,
            creativity_level=CreativityLevel.HYBRID,
            length_words=500,
        )
        assert spec.subject == "Marie Curie"
        assert spec.task == TaskName.INTERVIEW
        assert spec.creativity_level == CreativityLevel.HYBRID
        assert spec.length_words == 500

    def test_to_dict(self):
        """Should convert to dict correctly."""
        spec = PromptSpec(
            subject="Einstein",
            task=TaskName.NEWS_ARTICLE,
            creativity_level=CreativityLevel.FACTUAL,
        )
        d = spec.to_dict()
        assert d["subject"] == "Einstein"
        assert d["task"] == "NEWS_ARTICLE"
        assert d["creativity_level"] == "FACTUAL"

    def test_immutable(self):
        """PromptSpec should be immutable (frozen)."""
        spec = PromptSpec(
            subject="Test",
            task=TaskName.FAQ_HELP_PAGE,
            creativity_level=CreativityLevel.HYBRID,
        )
        with pytest.raises(AttributeError):
            spec.subject = "Changed"


class TestBannedPhrases:
    """Tests for banned phrases list."""

    def test_contains_key_terms(self):
        """Should contain key benchmark-related terms."""
        key_terms = ["wikipedia", "benchmark", "hallucination", "closed-book", "rag"]
        for term in key_terms:
            assert term in BANNED_PHRASES

    def test_check_banned_phrases_function(self):
        """check_banned_phrases should find violations."""
        text = "Please cite sources from Wikipedia for this benchmark evaluation."
        found = check_banned_phrases(text)
        assert "wikipedia" in found
        assert "benchmark" in found

    def test_clean_text_passes(self):
        """Clean text should pass the check."""
        text = "Tell me about the history of ancient Rome."
        found = check_banned_phrases(text)
        assert len(found) == 0


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_all_tasks(self):
        """Should return all tasks."""
        tasks = get_all_tasks()
        assert len(tasks) == 16
        assert TaskName.INTERVIEW in tasks

    def test_get_all_creativity_levels(self):
        """Should return all creativity levels."""
        levels = get_all_creativity_levels()
        assert len(levels) == 3

    def test_validate_task_name(self):
        """Should validate task names correctly."""
        assert validate_task_name("INTERVIEW") == TaskName.INTERVIEW
        assert validate_task_name("interview") == TaskName.INTERVIEW
        assert validate_task_name("INVALID") is None

    def test_validate_creativity_level(self):
        """Should validate creativity levels correctly."""
        assert validate_creativity_level("FACTUAL") == CreativityLevel.FACTUAL
        assert validate_creativity_level("hybrid") == CreativityLevel.HYBRID
        assert validate_creativity_level("INVALID") is None


# ==============================================================================
# Test Prompt Assemblers
# ==============================================================================


class TestMakeBlueprintPrompt:
    """Tests for blueprint prompt assembly."""

    def test_contains_subject(self):
        """Blueprint prompt should contain the subject."""
        prompt = make_blueprint_prompt("Marie Curie", "She was a scientist...")
        assert "Marie Curie" in prompt

    def test_contains_reference(self):
        """Blueprint prompt should contain the reference."""
        ref = "She discovered radium and polonium."
        prompt = make_blueprint_prompt("Marie Curie", ref)
        assert ref in prompt

    def test_contains_schema(self):
        """Blueprint prompt should contain JSON schema instructions."""
        prompt = make_blueprint_prompt("Test", "Reference text")
        assert "JSON" in prompt
        assert "scenario_hook" in prompt
        assert "audience" in prompt

    def test_contains_rules(self):
        """Blueprint prompt should contain critical rules."""
        prompt = make_blueprint_prompt("Test", "Reference text")
        assert "Do NOT quote" in prompt or "NOT quote" in prompt.upper()
        assert "Wikipedia" in prompt


class TestMakeNaturalizePrompt:
    """Tests for naturalize prompt assembly."""

    def test_contains_all_blocks(self):
        """Naturalize prompt should contain all lego blocks."""
        spec = PromptSpec(
            subject="Einstein",
            task=TaskName.INTERVIEW,
            creativity_level=CreativityLevel.HYBRID,
            length_words=500,
        )
        blueprint = {
            "scenario_hook": "A student needs help",
            "audience": "general readers",
            "intent": "learn about physics",
        }
        prompt = make_naturalize_prompt(spec, blueprint)

        # Should contain subject
        assert "Einstein" in prompt

        # Should contain task constraints
        assert "8 Q/A pairs" in prompt or "Q1:" in prompt

        # Should contain creativity style
        assert "engaging" in prompt.lower() or "creative" in prompt.lower()

        # Should contain length block
        assert "500" in prompt
        assert "words" in prompt.lower()

    def test_contains_blueprint(self):
        """Naturalize prompt should include blueprint JSON."""
        spec = PromptSpec(
            subject="Test",
            task=TaskName.FAQ_HELP_PAGE,
            creativity_level=CreativityLevel.FACTUAL,
        )
        blueprint = {"scenario_hook": "unique_scenario_text_12345"}
        prompt = make_naturalize_prompt(spec, blueprint)
        assert "unique_scenario_text_12345" in prompt


# ==============================================================================
# Test JSON Parsing
# ==============================================================================


class TestParseJsonResponse:
    """Tests for JSON response parsing."""

    def test_parse_clean_json(self):
        """Should parse clean JSON."""
        response = '{"key": "value", "number": 42}'
        result = parse_json_response(response)
        assert result == {"key": "value", "number": 42}

    def test_parse_json_in_code_block(self):
        """Should extract JSON from markdown code blocks."""
        response = """Here's the result:
```json
{"user_prompt": "Tell me about history"}
```
Hope this helps!"""
        result = parse_json_response(response)
        assert result is not None
        assert result["user_prompt"] == "Tell me about history"

    def test_parse_json_with_text_around(self):
        """Should find JSON embedded in other text."""
        response = 'Sure! {"answer": "test"} is what you need.'
        result = parse_json_response(response)
        assert result is not None
        assert result["answer"] == "test"

    def test_return_none_for_invalid(self):
        """Should return None for invalid JSON."""
        result = parse_json_response("This is not JSON at all")
        assert result is None

    def test_handle_empty_string(self):
        """Should handle empty string."""
        result = parse_json_response("")
        assert result is None

    def test_handle_none(self):
        """Should handle None input."""
        result = parse_json_response(None)
        assert result is None


class TestValidateBlueprint:
    """Tests for blueprint validation."""

    def test_valid_blueprint(self):
        """Should accept valid blueprint."""
        blueprint = {
            "scenario_hook": "A teacher wants to create materials",
            "audience": "high school students",
            "intent": "educational content",
            "constraints_soft": ["accessible", "engaging"],
            "seed_topics": ["history", "science"],
        }
        assert validate_blueprint(blueprint) is True

    def test_missing_required_field(self):
        """Should reject blueprint missing required fields."""
        blueprint = {
            "scenario_hook": "Some scenario",
            "audience": "readers",
            # missing "intent"
        }
        assert validate_blueprint(blueprint) is False

    def test_empty_required_field(self):
        """Should reject blueprint with empty required fields."""
        blueprint = {
            "scenario_hook": "   ",  # whitespace only
            "audience": "readers",
            "intent": "learn",
        }
        assert validate_blueprint(blueprint) is False


class TestValidateFinalPrompt:
    """Tests for final prompt validation."""

    def test_valid_prompt(self):
        """Should accept valid final prompt."""
        result = {
            "user_prompt": "Write an interview with Marie Curie about her research on radioactivity."
        }
        spec = PromptSpec(
            subject="Marie Curie",
            task=TaskName.INTERVIEW,
            creativity_level=CreativityLevel.HYBRID,
        )
        is_valid, issues = validate_final_prompt(result, spec)
        assert is_valid is True
        assert len(issues) == 0

    def test_missing_user_prompt(self):
        """Should reject result without user_prompt."""
        result = {"other_field": "value"}
        spec = PromptSpec(
            subject="Test",
            task=TaskName.FAQ_HELP_PAGE,
            creativity_level=CreativityLevel.FACTUAL,
        )
        is_valid, issues = validate_final_prompt(result, spec)
        assert is_valid is False
        assert "Missing 'user_prompt'" in issues[0]

    def test_banned_phrases_detected(self):
        """Should detect banned phrases in prompt."""
        result = {
            "user_prompt": "Use Wikipedia as your source for this benchmark test."
        }
        spec = PromptSpec(
            subject="Test",
            task=TaskName.FAQ_HELP_PAGE,
            creativity_level=CreativityLevel.FACTUAL,
        )
        is_valid, issues = validate_final_prompt(result, spec)
        assert is_valid is False
        assert any("banned" in issue.lower() for issue in issues)

    def test_subject_not_found(self):
        """Should detect when subject is missing from prompt."""
        result = {
            "user_prompt": "Tell me about something completely unrelated to anything."
        }
        spec = PromptSpec(
            subject="SpecificUniqueTopic12345",
            task=TaskName.INTERVIEW,
            creativity_level=CreativityLevel.HYBRID,
        )
        is_valid, issues = validate_final_prompt(result, spec)
        assert is_valid is False
        assert any("subject" in issue.lower() for issue in issues)


class TestReferenceLeakage:
    """Tests for reference leakage detection."""

    def test_no_leakage(self):
        """Should not detect leakage in clean prompt."""
        prompt = "Write about the history of science."
        reference = "Marie Curie discovered radium in 1898 in Paris."
        assert check_reference_leakage(prompt, reference) is False

    def test_detect_exact_copy(self):
        """Should detect when reference is copied exactly."""
        reference = "Marie Curie discovered radium and polonium through her research."
        prompt = f"Please write about this: {reference}"
        assert check_reference_leakage(prompt, reference) is True

    def test_detect_long_substring(self):
        """Should detect long common substrings."""
        reference = (
            "The scientist conducted groundbreaking experiments in radioactivity."
        )
        prompt = "Explain how the scientist conducted groundbreaking experiments in her laboratory."
        # The overlap "the scientist conducted groundbreaking experiments" is > 30 chars
        assert check_reference_leakage(prompt, reference) is True


# ==============================================================================
# Test GeneratedPrompt Dataclass
# ==============================================================================


class TestGeneratedPrompt:
    """Tests for GeneratedPrompt dataclass."""

    def test_creation(self):
        """Should create GeneratedPrompt correctly."""
        prompt = GeneratedPrompt(
            id="test-123",
            user_prompt="Write an article about physics.",
            reference_excerpt="Physics is the study of matter and energy.",
            metadata={"subject": "Physics", "task": "NEWS_ARTICLE"},
            quality_checks={"natural_sounding": True},
        )
        assert prompt.id == "test-123"
        assert prompt.user_prompt == "Write an article about physics."
        assert prompt.metadata["subject"] == "Physics"

    def test_to_dict(self):
        """Should convert to dict correctly."""
        prompt = GeneratedPrompt(
            id="test-456",
            user_prompt="Test prompt",
            reference_excerpt="Test reference",
            metadata={"key": "value"},
        )
        d = prompt.to_dict()
        assert d["id"] == "test-456"
        assert d["user_prompt"] == "Test prompt"
        assert d["reference_excerpt"] == "Test reference"
        assert d["metadata"]["key"] == "value"


# ==============================================================================
# Test Length Block Formatting
# ==============================================================================


class TestLengthBlock:
    """Tests for LENGTH_BLOCK template."""

    def test_format_length(self):
        """Should format with word count correctly."""
        result = LENGTH_BLOCK.format(length_words=500)
        assert "500" in result
        assert "words" in result.lower()
        assert "15%" in result or "15" in result


# ==============================================================================
# Integration-style tests (mock LLM calls)
# ==============================================================================


class TestPromptAssemblyIntegration:
    """Integration tests for prompt assembly."""

    def test_full_pipeline_prompt_structure(self):
        """Test that assembled prompts have correct structure for LLM."""
        # Blueprint prompt
        blueprint_prompt = make_blueprint_prompt(
            "Albert Einstein",
            "Albert Einstein was a theoretical physicist who developed the theory of relativity.",
        )

        # Check it's a well-formed instruction
        assert len(blueprint_prompt) > 200
        assert "Einstein" in blueprint_prompt
        assert "JSON" in blueprint_prompt

        # Naturalize prompt
        spec = PromptSpec(
            subject="Albert Einstein",
            task=TaskName.DOCUMENTARY_VOICEOVER,
            creativity_level=CreativityLevel.VERY_CREATIVE,
            length_words=600,
        )
        blueprint = {
            "scenario_hook": "A documentary filmmaker needs narration",
            "audience": "general viewers",
            "intent": "explain Einstein's contributions",
            "constraints_soft": ["inspiring", "accessible"],
            "seed_topics": ["relativity", "physics", "Nobel Prize"],
        }
        naturalize_prompt = make_naturalize_prompt(spec, blueprint)

        # Check it contains all necessary components
        assert "Einstein" in naturalize_prompt
        assert (
            "documentary" in naturalize_prompt.lower() or "SCENE" in naturalize_prompt
        )
        assert "600" in naturalize_prompt
        assert (
            "stylized" in naturalize_prompt.lower()
            or "creative" in naturalize_prompt.lower()
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
