"""Persona hub for generating diverse user personas at scale.

Generate and manage user persona profiles from combinatorial trait
definitions. Pure Python, no external dependencies, no LLM calls.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PersonaTrait:
    """A single trait axis with a category and possible values."""

    trait_name: str
    category: str  # "demographic", "expertise", "intent", "communication"
    values: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "trait_name": self.trait_name,
            "category": self.category,
            "values": list(self.values),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PersonaTrait:
        return cls(
            trait_name=data["trait_name"],
            category=data["category"],
            values=data.get("values", []),
        )


@dataclass
class PersonaProfile:
    """A single persona with assigned trait values."""

    persona_id: str
    name: str
    traits: dict[str, str] = field(default_factory=dict)  # trait_name -> value
    description: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "persona_id": self.persona_id,
            "name": self.name,
            "traits": dict(self.traits),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PersonaProfile:
        return cls(
            persona_id=data["persona_id"],
            name=data["name"],
            traits=data.get("traits", {}),
            description=data.get("description", ""),
        )


@dataclass
class PersonaHub:
    """A collection of persona profiles and their trait definitions."""

    profiles: list[PersonaProfile] = field(default_factory=list)
    trait_definitions: list[PersonaTrait] = field(default_factory=list)
    total_profiles: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "profiles": [p.as_dict() for p in self.profiles],
            "trait_definitions": [t.as_dict() for t in self.trait_definitions],
            "total_profiles": self.total_profiles,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PersonaHub:
        return cls(
            profiles=[
                PersonaProfile.from_dict(p) for p in data.get("profiles", [])
            ],
            trait_definitions=[
                PersonaTrait.from_dict(t)
                for t in data.get("trait_definitions", [])
            ],
            total_profiles=data.get("total_profiles", 0),
        )


DEFAULT_TRAITS: list[PersonaTrait] = [
    PersonaTrait(
        trait_name="demographic",
        category="demographic",
        values=[
            "student",
            "professional",
            "retiree",
            "researcher",
            "small_business_owner",
        ],
    ),
    PersonaTrait(
        trait_name="expertise",
        category="expertise",
        values=["novice", "intermediate", "advanced", "expert"],
    ),
    PersonaTrait(
        trait_name="intent",
        category="intent",
        values=[
            "informational",
            "transactional",
            "navigational",
            "troubleshooting",
            "exploratory",
        ],
    ),
    PersonaTrait(
        trait_name="communication",
        category="communication",
        values=["formal", "casual", "technical", "terse", "verbose"],
    ),
]


def _build_persona_name(traits: dict[str, str]) -> str:
    """Build a readable name from trait values."""
    parts: list[str] = []
    for key in sorted(traits.keys()):
        parts.append(traits[key])
    return "_".join(parts)


def generate_persona(
    traits: list[PersonaTrait],
    rng: random.Random | None = None,
) -> PersonaProfile:
    """Randomly combine one value from each trait into a persona."""
    if rng is None:
        rng = random.Random()
    chosen: dict[str, str] = {}
    for trait in traits:
        if trait.values:
            chosen[trait.trait_name] = rng.choice(trait.values)
    name = _build_persona_name(chosen)
    persona_id = str(uuid.uuid4())
    desc_parts = [f"{k}={v}" for k, v in sorted(chosen.items())]
    description = ", ".join(desc_parts)
    return PersonaProfile(
        persona_id=persona_id,
        name=name,
        traits=chosen,
        description=description,
    )


def generate_persona_hub(
    n_personas: int = 20,
    traits: list[PersonaTrait] | None = None,
    seed: int | None = None,
) -> PersonaHub:
    """Generate n diverse personas ensuring coverage across trait categories.

    Uses round-robin value selection per trait to maximize diversity before
    falling back to random sampling for the remainder.
    """
    if traits is None:
        traits = list(DEFAULT_TRAITS)
    rng = random.Random(seed)

    # First pass: ensure every trait value appears at least once
    # by cycling through values for each trait
    value_queues: dict[str, list[str]] = {}
    for trait in traits:
        if trait.values:
            shuffled = list(trait.values)
            rng.shuffle(shuffled)
            value_queues[trait.trait_name] = shuffled

    profiles: list[PersonaProfile] = []
    seen_combos: set[tuple[str, ...]] = set()

    for i in range(n_personas):
        chosen: dict[str, str] = {}
        for trait in traits:
            if not trait.values:
                continue
            queue = value_queues[trait.trait_name]
            if not queue:
                # Refill and reshuffle
                queue = list(trait.values)
                rng.shuffle(queue)
                value_queues[trait.trait_name] = queue
            chosen[trait.trait_name] = queue.pop()

        combo = tuple(chosen.get(t.trait_name, "") for t in traits)
        # Allow duplicates if we exhaust unique combos
        if combo in seen_combos and i < n_personas:
            # Try one more time with pure random
            for trait in traits:
                if trait.values:
                    chosen[trait.trait_name] = rng.choice(trait.values)
            combo = tuple(chosen.get(t.trait_name, "") for t in traits)
        seen_combos.add(combo)

        name = _build_persona_name(chosen)
        persona_id = str(uuid.uuid4())
        desc_parts = [f"{k}={v}" for k, v in sorted(chosen.items())]
        description = ", ".join(desc_parts)
        profiles.append(
            PersonaProfile(
                persona_id=persona_id,
                name=name,
                traits=chosen,
                description=description,
            )
        )

    return PersonaHub(
        profiles=profiles,
        trait_definitions=traits,
        total_profiles=len(profiles),
    )


def expand_persona_pairs(
    hub: PersonaHub,
    n_pairs: int = 5,
) -> list[tuple[PersonaProfile, PersonaProfile]]:
    """Create interesting persona pairs for multi-turn simulation.

    Prioritizes pairs with opposite expertise levels or different intents.
    Falls back to random pairing if traits are not available.
    """
    if len(hub.profiles) < 2:
        return []

    # Score each pair by how "interesting" the contrast is
    scored_pairs: list[tuple[float, int, int]] = []
    expertise_order = {"novice": 0, "intermediate": 1, "advanced": 2, "expert": 3}

    for i in range(len(hub.profiles)):
        for j in range(i + 1, len(hub.profiles)):
            a = hub.profiles[i]
            b = hub.profiles[j]
            score = 0.0

            # Reward expertise gap
            ea = expertise_order.get(a.traits.get("expertise", ""), -1)
            eb = expertise_order.get(b.traits.get("expertise", ""), -1)
            if ea >= 0 and eb >= 0:
                score += abs(ea - eb)

            # Reward different intents
            if a.traits.get("intent") != b.traits.get("intent"):
                score += 1.0

            # Reward different communication styles
            if a.traits.get("communication") != b.traits.get("communication"):
                score += 0.5

            # Reward different demographics
            if a.traits.get("demographic") != b.traits.get("demographic"):
                score += 0.5

            scored_pairs.append((score, i, j))

    # Sort by score descending
    scored_pairs.sort(key=lambda x: x[0], reverse=True)

    pairs: list[tuple[PersonaProfile, PersonaProfile]] = []
    used: set[int] = set()
    for _score, i, j in scored_pairs:
        if len(pairs) >= n_pairs:
            break
        if i not in used and j not in used:
            pairs.append((hub.profiles[i], hub.profiles[j]))
            used.add(i)
            used.add(j)

    # If we still need more pairs, relax the uniqueness constraint
    if len(pairs) < n_pairs:
        for _score, i, j in scored_pairs:
            if len(pairs) >= n_pairs:
                break
            pair = (hub.profiles[i], hub.profiles[j])
            if pair not in pairs:
                pairs.append(pair)

    return pairs[:n_pairs]


def check_diversity_coverage(hub: PersonaHub) -> dict[str, float]:
    """Per-trait-category coverage: fraction of values represented."""
    coverage: dict[str, float] = {}
    for trait in hub.trait_definitions:
        if not trait.values:
            coverage[trait.category] = 1.0
            continue
        seen: set[str] = set()
        for profile in hub.profiles:
            val = profile.traits.get(trait.trait_name)
            if val is not None:
                seen.add(val)
        coverage[trait.category] = len(seen) / len(trait.values)
    return coverage


def filter_personas(
    hub: PersonaHub,
    trait_name: str,
    value: str,
) -> list[PersonaProfile]:
    """Filter personas by a specific trait value."""
    return [
        p for p in hub.profiles if p.traits.get(trait_name) == value
    ]


def format_persona_hub(hub: PersonaHub) -> str:
    """Format a human-readable persona catalog."""
    lines: list[str] = []
    lines.append("Persona Hub Catalog")
    lines.append("=" * 40)
    lines.append(f"Total profiles: {hub.total_profiles}")
    lines.append(f"Trait definitions: {len(hub.trait_definitions)}")
    lines.append("")

    # Trait definitions
    lines.append("Trait Definitions:")
    for trait in hub.trait_definitions:
        lines.append(
            f"  {trait.trait_name} ({trait.category}): "
            f"{', '.join(trait.values)}"
        )
    lines.append("")

    # Coverage
    coverage = check_diversity_coverage(hub)
    lines.append("Diversity Coverage:")
    for category, frac in sorted(coverage.items()):
        lines.append(f"  {category}: {frac:.0%}")
    lines.append("")

    # Profiles
    lines.append("Profiles:")
    for p in hub.profiles:
        lines.append(f"  [{p.persona_id[:8]}] {p.name}")
        if p.description:
            lines.append(f"    {p.description}")

    return "\n".join(lines)
