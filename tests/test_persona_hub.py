"""Tests for persona hub module."""

from __future__ import annotations

from evalyn_sdk.persona_hub import (
    DEFAULT_TRAITS,
    PersonaHub,
    PersonaProfile,
    PersonaTrait,
    check_diversity_coverage,
    expand_persona_pairs,
    filter_personas,
    format_persona_hub,
    generate_persona,
    generate_persona_hub,
)


# ---------------------------------------------------------------------------
# PersonaTrait
# ---------------------------------------------------------------------------


class TestPersonaTrait:
    def test_basic_creation(self):
        t = PersonaTrait(
            trait_name="expertise",
            category="expertise",
            values=["novice", "expert"],
        )
        assert t.trait_name == "expertise"
        assert t.category == "expertise"
        assert len(t.values) == 2

    def test_default_values(self):
        t = PersonaTrait(trait_name="test", category="test")
        assert t.values == []

    def test_as_dict(self):
        t = PersonaTrait(
            trait_name="intent",
            category="intent",
            values=["a", "b"],
        )
        d = t.as_dict()
        assert d["trait_name"] == "intent"
        assert d["values"] == ["a", "b"]

    def test_from_dict(self):
        d = {"trait_name": "comm", "category": "communication", "values": ["formal"]}
        t = PersonaTrait.from_dict(d)
        assert t.trait_name == "comm"
        assert t.category == "communication"

    def test_from_dict_missing_values(self):
        d = {"trait_name": "x", "category": "y"}
        t = PersonaTrait.from_dict(d)
        assert t.values == []

    def test_roundtrip(self):
        original = PersonaTrait(
            trait_name="demo", category="demographic", values=["a", "b", "c"]
        )
        restored = PersonaTrait.from_dict(original.as_dict())
        assert restored.trait_name == original.trait_name
        assert restored.values == original.values


# ---------------------------------------------------------------------------
# PersonaProfile
# ---------------------------------------------------------------------------


class TestPersonaProfile:
    def test_basic_creation(self):
        p = PersonaProfile(
            persona_id="p1",
            name="test_persona",
            traits={"expertise": "novice"},
        )
        assert p.persona_id == "p1"
        assert p.traits["expertise"] == "novice"

    def test_default_description(self):
        p = PersonaProfile(persona_id="p2", name="test")
        assert p.description == ""

    def test_default_traits(self):
        p = PersonaProfile(persona_id="p3", name="test")
        assert p.traits == {}

    def test_as_dict(self):
        p = PersonaProfile(
            persona_id="p4",
            name="named",
            traits={"a": "b"},
            description="desc",
        )
        d = p.as_dict()
        assert d["persona_id"] == "p4"
        assert d["traits"] == {"a": "b"}
        assert d["description"] == "desc"

    def test_from_dict(self):
        d = {
            "persona_id": "p5",
            "name": "from_dict",
            "traits": {"x": "y"},
            "description": "hello",
        }
        p = PersonaProfile.from_dict(d)
        assert p.persona_id == "p5"
        assert p.traits["x"] == "y"

    def test_from_dict_missing_optionals(self):
        d = {"persona_id": "p6", "name": "minimal"}
        p = PersonaProfile.from_dict(d)
        assert p.traits == {}
        assert p.description == ""

    def test_roundtrip(self):
        original = PersonaProfile(
            persona_id="rt",
            name="roundtrip",
            traits={"a": "1", "b": "2"},
            description="test roundtrip",
        )
        restored = PersonaProfile.from_dict(original.as_dict())
        assert restored.persona_id == original.persona_id
        assert restored.traits == original.traits


# ---------------------------------------------------------------------------
# PersonaHub
# ---------------------------------------------------------------------------


class TestPersonaHub:
    def test_defaults(self):
        h = PersonaHub()
        assert h.profiles == []
        assert h.trait_definitions == []
        assert h.total_profiles == 0

    def test_as_dict(self):
        p = PersonaProfile(persona_id="p1", name="n")
        t = PersonaTrait(trait_name="t", category="c", values=["v"])
        h = PersonaHub(profiles=[p], trait_definitions=[t], total_profiles=1)
        d = h.as_dict()
        assert len(d["profiles"]) == 1
        assert len(d["trait_definitions"]) == 1

    def test_from_dict(self):
        d = {
            "profiles": [{"persona_id": "p1", "name": "n", "traits": {}}],
            "trait_definitions": [
                {"trait_name": "t", "category": "c", "values": ["v"]}
            ],
            "total_profiles": 1,
        }
        h = PersonaHub.from_dict(d)
        assert len(h.profiles) == 1
        assert h.total_profiles == 1

    def test_roundtrip(self):
        hub = generate_persona_hub(n_personas=5, seed=42)
        restored = PersonaHub.from_dict(hub.as_dict())
        assert restored.total_profiles == hub.total_profiles
        assert len(restored.profiles) == len(hub.profiles)


# ---------------------------------------------------------------------------
# DEFAULT_TRAITS
# ---------------------------------------------------------------------------


class TestDefaultTraits:
    def test_four_categories(self):
        categories = {t.category for t in DEFAULT_TRAITS}
        assert categories == {"demographic", "expertise", "intent", "communication"}

    def test_demographic_values(self):
        demo = [t for t in DEFAULT_TRAITS if t.category == "demographic"][0]
        assert "student" in demo.values
        assert "professional" in demo.values
        assert len(demo.values) == 5

    def test_expertise_values(self):
        exp = [t for t in DEFAULT_TRAITS if t.category == "expertise"][0]
        assert "novice" in exp.values
        assert "expert" in exp.values
        assert len(exp.values) == 4

    def test_intent_values(self):
        intent = [t for t in DEFAULT_TRAITS if t.category == "intent"][0]
        assert "informational" in intent.values
        assert len(intent.values) == 5

    def test_communication_values(self):
        comm = [t for t in DEFAULT_TRAITS if t.category == "communication"][0]
        assert "formal" in comm.values
        assert len(comm.values) == 5


# ---------------------------------------------------------------------------
# generate_persona
# ---------------------------------------------------------------------------


class TestGeneratePersona:
    def test_returns_persona_profile(self):
        p = generate_persona(DEFAULT_TRAITS)
        assert isinstance(p, PersonaProfile)

    def test_has_all_traits(self):
        p = generate_persona(DEFAULT_TRAITS)
        for trait in DEFAULT_TRAITS:
            assert trait.trait_name in p.traits

    def test_trait_values_from_definitions(self):
        p = generate_persona(DEFAULT_TRAITS)
        for trait in DEFAULT_TRAITS:
            assert p.traits[trait.trait_name] in trait.values

    def test_has_persona_id(self):
        p = generate_persona(DEFAULT_TRAITS)
        assert len(p.persona_id) > 0

    def test_has_name(self):
        p = generate_persona(DEFAULT_TRAITS)
        assert len(p.name) > 0

    def test_has_description(self):
        p = generate_persona(DEFAULT_TRAITS)
        assert len(p.description) > 0

    def test_deterministic_with_rng(self):
        import random

        p1 = generate_persona(DEFAULT_TRAITS, rng=random.Random(42))
        p2 = generate_persona(DEFAULT_TRAITS, rng=random.Random(42))
        assert p1.traits == p2.traits

    def test_empty_traits(self):
        p = generate_persona([])
        assert p.traits == {}


# ---------------------------------------------------------------------------
# generate_persona_hub
# ---------------------------------------------------------------------------


class TestGeneratePersonaHub:
    def test_returns_hub(self):
        hub = generate_persona_hub(n_personas=5, seed=1)
        assert isinstance(hub, PersonaHub)

    def test_correct_count(self):
        hub = generate_persona_hub(n_personas=10, seed=1)
        assert len(hub.profiles) == 10
        assert hub.total_profiles == 10

    def test_default_count(self):
        hub = generate_persona_hub(seed=1)
        assert len(hub.profiles) == 20

    def test_deterministic_with_seed(self):
        h1 = generate_persona_hub(n_personas=5, seed=42)
        h2 = generate_persona_hub(n_personas=5, seed=42)
        traits1 = [p.traits for p in h1.profiles]
        traits2 = [p.traits for p in h2.profiles]
        assert traits1 == traits2

    def test_different_seeds_differ(self):
        h1 = generate_persona_hub(n_personas=10, seed=1)
        h2 = generate_persona_hub(n_personas=10, seed=2)
        traits1 = [p.traits for p in h1.profiles]
        traits2 = [p.traits for p in h2.profiles]
        assert traits1 != traits2

    def test_uses_default_traits(self):
        hub = generate_persona_hub(n_personas=5, seed=1)
        assert len(hub.trait_definitions) == len(DEFAULT_TRAITS)

    def test_custom_traits(self):
        custom = [
            PersonaTrait(trait_name="mood", category="mood", values=["happy", "sad"])
        ]
        hub = generate_persona_hub(n_personas=3, traits=custom, seed=1)
        assert len(hub.trait_definitions) == 1
        for p in hub.profiles:
            assert "mood" in p.traits
            assert p.traits["mood"] in ["happy", "sad"]

    def test_diversity_with_enough_personas(self):
        # With 20 personas and 4-5 values per trait, coverage should be good
        hub = generate_persona_hub(n_personas=50, seed=42)
        coverage = check_diversity_coverage(hub)
        for cat, frac in coverage.items():
            assert frac >= 0.8, f"Low coverage for {cat}: {frac}"


# ---------------------------------------------------------------------------
# expand_persona_pairs
# ---------------------------------------------------------------------------


class TestExpandPersonaPairs:
    def test_returns_pairs(self):
        hub = generate_persona_hub(n_personas=10, seed=1)
        pairs = expand_persona_pairs(hub, n_pairs=3)
        assert len(pairs) == 3
        for a, b in pairs:
            assert isinstance(a, PersonaProfile)
            assert isinstance(b, PersonaProfile)

    def test_too_few_profiles(self):
        hub = PersonaHub(profiles=[], trait_definitions=[], total_profiles=0)
        assert expand_persona_pairs(hub) == []

    def test_single_profile(self):
        p = PersonaProfile(persona_id="only", name="solo", traits={})
        hub = PersonaHub(profiles=[p], trait_definitions=[], total_profiles=1)
        assert expand_persona_pairs(hub) == []

    def test_pairs_prefer_diversity(self):
        hub = generate_persona_hub(n_personas=20, seed=42)
        pairs = expand_persona_pairs(hub, n_pairs=5)
        # At least some pairs should have different expertise levels
        expertise_diffs = 0
        for a, b in pairs:
            if a.traits.get("expertise") != b.traits.get("expertise"):
                expertise_diffs += 1
        assert expertise_diffs >= 1

    def test_max_pairs_capped(self):
        hub = generate_persona_hub(n_personas=4, seed=1)
        # Maximum possible unique pairs from 4 profiles = C(4,2) = 6
        pairs = expand_persona_pairs(hub, n_pairs=10)
        assert len(pairs) <= 10


# ---------------------------------------------------------------------------
# check_diversity_coverage
# ---------------------------------------------------------------------------


class TestDiversityCoverage:
    def test_full_coverage(self):
        hub = generate_persona_hub(n_personas=100, seed=42)
        coverage = check_diversity_coverage(hub)
        for cat, frac in coverage.items():
            assert frac == 1.0, f"{cat} not fully covered: {frac}"

    def test_empty_hub(self):
        hub = PersonaHub(
            profiles=[],
            trait_definitions=DEFAULT_TRAITS,
            total_profiles=0,
        )
        coverage = check_diversity_coverage(hub)
        for cat, frac in coverage.items():
            assert frac == 0.0

    def test_single_persona_partial(self):
        p = PersonaProfile(
            persona_id="x",
            name="x",
            traits={
                "demographic": "student",
                "expertise": "novice",
                "intent": "informational",
                "communication": "formal",
            },
        )
        hub = PersonaHub(
            profiles=[p],
            trait_definitions=DEFAULT_TRAITS,
            total_profiles=1,
        )
        coverage = check_diversity_coverage(hub)
        # Each category has 1 out of N values covered
        assert coverage["demographic"] == 1 / 5
        assert coverage["expertise"] == 1 / 4
        assert coverage["intent"] == 1 / 5
        assert coverage["communication"] == 1 / 5

    def test_empty_trait_values(self):
        t = PersonaTrait(trait_name="empty", category="empty", values=[])
        hub = PersonaHub(
            profiles=[],
            trait_definitions=[t],
            total_profiles=0,
        )
        coverage = check_diversity_coverage(hub)
        assert coverage["empty"] == 1.0


# ---------------------------------------------------------------------------
# filter_personas
# ---------------------------------------------------------------------------


class TestFilterPersonas:
    def test_filter_by_expertise(self):
        hub = generate_persona_hub(n_personas=50, seed=42)
        experts = filter_personas(hub, "expertise", "expert")
        for p in experts:
            assert p.traits["expertise"] == "expert"
        assert len(experts) > 0

    def test_filter_no_match(self):
        hub = generate_persona_hub(n_personas=5, seed=42)
        result = filter_personas(hub, "expertise", "nonexistent_value")
        assert result == []

    def test_filter_nonexistent_trait(self):
        hub = generate_persona_hub(n_personas=5, seed=42)
        result = filter_personas(hub, "nonexistent_trait", "value")
        assert result == []

    def test_filter_returns_subset(self):
        hub = generate_persona_hub(n_personas=20, seed=42)
        students = filter_personas(hub, "demographic", "student")
        assert len(students) <= 20

    def test_filter_preserves_profile(self):
        hub = generate_persona_hub(n_personas=20, seed=42)
        novices = filter_personas(hub, "expertise", "novice")
        for p in novices:
            assert isinstance(p, PersonaProfile)
            assert p.persona_id
            assert p.name


# ---------------------------------------------------------------------------
# format_persona_hub
# ---------------------------------------------------------------------------


class TestFormatPersonaHub:
    def test_contains_header(self):
        hub = generate_persona_hub(n_personas=3, seed=1)
        report = format_persona_hub(hub)
        assert "Persona Hub Catalog" in report

    def test_contains_total(self):
        hub = generate_persona_hub(n_personas=5, seed=1)
        report = format_persona_hub(hub)
        assert "Total profiles: 5" in report

    def test_contains_trait_definitions(self):
        hub = generate_persona_hub(n_personas=3, seed=1)
        report = format_persona_hub(hub)
        assert "Trait Definitions:" in report
        assert "demographic" in report

    def test_contains_coverage(self):
        hub = generate_persona_hub(n_personas=3, seed=1)
        report = format_persona_hub(hub)
        assert "Diversity Coverage:" in report

    def test_contains_profiles(self):
        hub = generate_persona_hub(n_personas=2, seed=1)
        report = format_persona_hub(hub)
        assert "Profiles:" in report
        # Each profile should show its truncated ID
        for p in hub.profiles:
            assert p.persona_id[:8] in report

    def test_empty_hub(self):
        hub = PersonaHub()
        report = format_persona_hub(hub)
        assert "Total profiles: 0" in report
