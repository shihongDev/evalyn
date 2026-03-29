"""Tests for persona-based simulation module."""

from __future__ import annotations

import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.persona_simulation import (
    BUILTIN_PERSONAS,
    Persona,
    PersonaTemplate,
    SimulatedInput,
    apply_persona_style,
    filter_by_persona,
    format_persona_catalog,
    generate_cohort,
    generate_inputs_for_persona,
    generate_persona_prompt,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_persona(**overrides) -> Persona:
    defaults = {
        "persona_id": "test",
        "name": "Test User",
        "description": "A test persona",
        "traits": ["curious"],
        "language_style": "casual",
        "expertise_level": "intermediate",
        "custom_instructions": "",
    }
    defaults.update(overrides)
    return Persona(**defaults)


# ---------------------------------------------------------------------------
# Persona dataclass
# ---------------------------------------------------------------------------


class TestPersona:
    def test_as_dict_roundtrip(self):
        p = _make_persona()
        d = p.as_dict()
        p2 = Persona.from_dict(d)
        assert p2.persona_id == p.persona_id
        assert p2.name == p.name
        assert p2.traits == p.traits

    def test_from_dict_defaults(self):
        p = Persona.from_dict({
            "persona_id": "x",
            "name": "X",
            "description": "desc",
        })
        assert p.language_style == "casual"
        assert p.expertise_level == "intermediate"
        assert p.custom_instructions == ""
        assert p.traits == []

    def test_custom_instructions_preserved(self):
        p = _make_persona(custom_instructions="always respond in haiku")
        d = p.as_dict()
        assert d["custom_instructions"] == "always respond in haiku"
        p2 = Persona.from_dict(d)
        assert p2.custom_instructions == "always respond in haiku"

    def test_traits_are_copied(self):
        original_traits = ["a", "b"]
        p = _make_persona(traits=original_traits)
        d = p.as_dict()
        d["traits"].append("c")
        assert len(p.traits) == 2  # original unchanged


# ---------------------------------------------------------------------------
# PersonaTemplate dataclass
# ---------------------------------------------------------------------------


class TestPersonaTemplate:
    def test_as_dict_roundtrip(self):
        t = PersonaTemplate(
            template_id="t1",
            text_template="As {persona_name}, ask about {topic}",
            persona_id="novice",
        )
        d = t.as_dict()
        t2 = PersonaTemplate.from_dict(d)
        assert t2.template_id == "t1"
        assert t2.text_template == t.text_template
        assert t2.persona_id == "novice"

    def test_template_has_all_fields(self):
        t = PersonaTemplate(
            template_id="t2",
            text_template="{persona_name} ({trait_list}) [{style}]: {topic}",
            persona_id="power_user",
        )
        d = t.as_dict()
        assert "template_id" in d
        assert "text_template" in d
        assert "persona_id" in d


# ---------------------------------------------------------------------------
# SimulatedInput dataclass
# ---------------------------------------------------------------------------


class TestSimulatedInput:
    def test_as_dict_roundtrip(self):
        si = SimulatedInput(
            input_text="hello",
            persona_id="novice",
            persona_name="Novice User",
            tags=["novice", "simple"],
            generated_at="2026-01-01T00:00:00",
        )
        d = si.as_dict()
        si2 = SimulatedInput.from_dict(d)
        assert si2.input_text == "hello"
        assert si2.tags == ["novice", "simple"]

    def test_from_dict_defaults(self):
        si = SimulatedInput.from_dict({
            "input_text": "hi",
            "persona_id": "x",
            "persona_name": "X",
        })
        assert si.tags == []
        assert si.generated_at == ""

    def test_tags_copied(self):
        si = SimulatedInput(
            input_text="hi",
            persona_id="x",
            persona_name="X",
            tags=["a"],
            generated_at="now",
        )
        d = si.as_dict()
        d["tags"].append("b")
        assert len(si.tags) == 1


# ---------------------------------------------------------------------------
# BUILTIN_PERSONAS
# ---------------------------------------------------------------------------


class TestBuiltinPersonas:
    def test_has_five_personas(self):
        assert len(BUILTIN_PERSONAS) == 5

    def test_expected_keys(self):
        expected = {"novice", "power_user", "adversarial", "non_native", "enterprise"}
        assert set(BUILTIN_PERSONAS.keys()) == expected

    def test_novice_style(self):
        p = BUILTIN_PERSONAS["novice"]
        assert p.language_style == "simple"
        assert p.expertise_level == "novice"

    def test_power_user_style(self):
        p = BUILTIN_PERSONAS["power_user"]
        assert p.language_style == "technical"
        assert p.expertise_level == "expert"

    def test_adversarial_style(self):
        p = BUILTIN_PERSONAS["adversarial"]
        assert p.expertise_level == "adversarial"

    def test_non_native_style(self):
        p = BUILTIN_PERSONAS["non_native"]
        assert p.language_style == "simple"
        assert p.expertise_level == "intermediate"

    def test_enterprise_style(self):
        p = BUILTIN_PERSONAS["enterprise"]
        assert p.language_style == "formal"
        assert p.expertise_level == "expert"

    def test_all_have_descriptions(self):
        for pid, persona in BUILTIN_PERSONAS.items():
            assert persona.description, f"{pid} missing description"

    def test_all_have_traits(self):
        for pid, persona in BUILTIN_PERSONAS.items():
            assert len(persona.traits) > 0, f"{pid} has no traits"


# ---------------------------------------------------------------------------
# generate_persona_prompt
# ---------------------------------------------------------------------------


class TestGeneratePersonaPrompt:
    def test_contains_persona_name(self):
        p = BUILTIN_PERSONAS["novice"]
        prompt = generate_persona_prompt(p, "Python basics")
        assert "Novice User" in prompt

    def test_contains_topic(self):
        p = BUILTIN_PERSONAS["power_user"]
        prompt = generate_persona_prompt(p, "API design")
        assert "API design" in prompt

    def test_contains_traits(self):
        p = _make_persona(traits=["detail-oriented", "skeptical"])
        prompt = generate_persona_prompt(p, "testing")
        assert "detail-oriented" in prompt
        assert "skeptical" in prompt

    def test_contains_custom_instructions(self):
        p = _make_persona(custom_instructions="respond in bullet points")
        prompt = generate_persona_prompt(p, "anything")
        assert "respond in bullet points" in prompt

    def test_no_custom_instructions_omits_line(self):
        p = _make_persona(custom_instructions="")
        prompt = generate_persona_prompt(p, "anything")
        assert "Additional instructions" not in prompt

    def test_contains_style_and_level(self):
        p = _make_persona(language_style="formal", expertise_level="expert")
        prompt = generate_persona_prompt(p, "topic")
        assert "formal" in prompt
        assert "expert" in prompt


# ---------------------------------------------------------------------------
# apply_persona_style
# ---------------------------------------------------------------------------


class TestApplyPersonaStyle:
    def test_adversarial_uppercases(self):
        p = _make_persona(expertise_level="adversarial")
        result = apply_persona_style("hello world", p)
        assert result == "HELLO WORLD"

    def test_novice_adds_explain_prefix(self):
        p = _make_persona(expertise_level="novice", language_style="casual")
        result = apply_persona_style("machine learning", p)
        assert result.startswith("Can you explain")
        assert result.endswith("?")
        assert "machine learning" in result

    def test_simple_style_adds_explain_prefix(self):
        p = _make_persona(expertise_level="intermediate", language_style="simple")
        result = apply_persona_style("databases", p)
        assert result.startswith("Can you explain")

    def test_formal_adds_qualifier(self):
        p = _make_persona(expertise_level="expert", language_style="formal")
        result = apply_persona_style("compliance rules", p)
        assert "Per our requirements" in result
        assert "compliance rules" in result

    def test_technical_returns_unchanged(self):
        p = _make_persona(expertise_level="expert", language_style="technical")
        result = apply_persona_style("SELECT * FROM users", p)
        assert result == "SELECT * FROM users"

    def test_intermediate_casual_returns_unchanged(self):
        p = _make_persona(expertise_level="intermediate", language_style="casual")
        result = apply_persona_style("hello", p)
        assert result == "hello"

    def test_novice_strips_trailing_punctuation(self):
        p = _make_persona(expertise_level="novice")
        result = apply_persona_style("what is AI?", p)
        assert not result.endswith("??")
        assert result.endswith("?")


# ---------------------------------------------------------------------------
# generate_inputs_for_persona
# ---------------------------------------------------------------------------


class TestGenerateInputsForPersona:
    def test_one_input_per_topic(self):
        p = BUILTIN_PERSONAS["novice"]
        topics = ["Python", "JavaScript", "Rust"]
        inputs = generate_inputs_for_persona(p, topics)
        assert len(inputs) == 3

    def test_persona_id_on_all(self):
        p = BUILTIN_PERSONAS["power_user"]
        inputs = generate_inputs_for_persona(p, ["topic1"])
        assert all(i.persona_id == "power_user" for i in inputs)

    def test_persona_name_on_all(self):
        p = BUILTIN_PERSONAS["enterprise"]
        inputs = generate_inputs_for_persona(p, ["governance"])
        assert inputs[0].persona_name == "Enterprise User"

    def test_tags_include_style_and_level(self):
        p = BUILTIN_PERSONAS["novice"]
        inputs = generate_inputs_for_persona(p, ["topic"])
        assert "novice" in inputs[0].tags
        assert "simple" in inputs[0].tags

    def test_generated_at_is_set(self):
        p = _make_persona()
        inputs = generate_inputs_for_persona(p, ["t"])
        assert inputs[0].generated_at != ""

    def test_with_template(self):
        p = BUILTIN_PERSONAS["novice"]
        tmpl = PersonaTemplate(
            template_id="t1",
            text_template="[{persona_name}] ({style}) asks about {topic}",
            persona_id="novice",
        )
        inputs = generate_inputs_for_persona(p, ["ML"], template=tmpl)
        assert "[Novice User]" in inputs[0].input_text
        assert "ML" in inputs[0].input_text
        assert "(simple)" in inputs[0].input_text

    def test_template_with_trait_list(self):
        p = _make_persona(traits=["curious", "patient"])
        tmpl = PersonaTemplate(
            template_id="t2",
            text_template="Traits: {trait_list} | Topic: {topic}",
            persona_id="test",
        )
        inputs = generate_inputs_for_persona(p, ["AI"], template=tmpl)
        assert "curious, patient" in inputs[0].input_text

    def test_empty_topics_returns_empty(self):
        p = _make_persona()
        inputs = generate_inputs_for_persona(p, [])
        assert inputs == []

    def test_without_template_applies_style(self):
        p = BUILTIN_PERSONAS["adversarial"]
        inputs = generate_inputs_for_persona(p, ["hack this"])
        assert inputs[0].input_text == "HACK THIS"


# ---------------------------------------------------------------------------
# generate_cohort
# ---------------------------------------------------------------------------


class TestGenerateCohort:
    def test_count_is_personas_times_topics(self):
        personas = [BUILTIN_PERSONAS["novice"], BUILTIN_PERSONAS["power_user"]]
        topics = ["A", "B", "C"]
        cohort = generate_cohort(personas, topics)
        assert len(cohort) == 6

    def test_all_personas_represented(self):
        personas = list(BUILTIN_PERSONAS.values())
        topics = ["topic"]
        cohort = generate_cohort(personas, topics)
        ids = {i.persona_id for i in cohort}
        assert ids == set(BUILTIN_PERSONAS.keys())

    def test_empty_personas_returns_empty(self):
        assert generate_cohort([], ["t"]) == []

    def test_empty_topics_returns_empty(self):
        assert generate_cohort([_make_persona()], []) == []


# ---------------------------------------------------------------------------
# filter_by_persona
# ---------------------------------------------------------------------------


class TestFilterByPersona:
    def test_filters_correctly(self):
        inputs = [
            SimulatedInput("a", "novice", "N", [], "now"),
            SimulatedInput("b", "expert", "E", [], "now"),
            SimulatedInput("c", "novice", "N", [], "now"),
        ]
        result = filter_by_persona(inputs, "novice")
        assert len(result) == 2
        assert all(i.persona_id == "novice" for i in result)

    def test_no_match_returns_empty(self):
        inputs = [SimulatedInput("a", "novice", "N", [], "now")]
        result = filter_by_persona(inputs, "expert")
        assert result == []

    def test_empty_input_returns_empty(self):
        result = filter_by_persona([], "any")
        assert result == []


# ---------------------------------------------------------------------------
# format_persona_catalog
# ---------------------------------------------------------------------------


class TestFormatPersonaCatalog:
    def test_contains_all_persona_ids(self):
        catalog = format_persona_catalog(BUILTIN_PERSONAS)
        for pid in BUILTIN_PERSONAS:
            assert pid in catalog

    def test_contains_names(self):
        catalog = format_persona_catalog(BUILTIN_PERSONAS)
        for persona in BUILTIN_PERSONAS.values():
            assert persona.name in catalog

    def test_contains_descriptions(self):
        catalog = format_persona_catalog(BUILTIN_PERSONAS)
        for persona in BUILTIN_PERSONAS.values():
            assert persona.description in catalog

    def test_empty_dict_returns_empty_string(self):
        assert format_persona_catalog({}) == ""

    def test_single_persona(self):
        personas = {"test": _make_persona()}
        catalog = format_persona_catalog(personas)
        assert "test:" in catalog
        assert "Test User" in catalog
