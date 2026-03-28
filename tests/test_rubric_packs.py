"""Tests for domain-specific rubric packs (rubric_packs module)."""

from __future__ import annotations

import pytest

from evalyn_sdk.rubric_packs import (
    BUILTIN_PACKS,
    PackRegistry,
    RubricPack,
    format_pack_detail,
    format_pack_listing,
)


# ---------------------------------------------------------------------------
# RubricPack dataclass
# ---------------------------------------------------------------------------


class TestRubricPack:
    def test_creation(self):
        pack = RubricPack(
            pack_id="test",
            domain="testing",
            description="Test pack",
            rubrics=[{"id": "test/r1", "type": "quality"}],
            version="1.0.0",
            author="tester",
        )
        assert pack.pack_id == "test"
        assert pack.domain == "testing"
        assert len(pack.rubrics) == 1

    def test_as_dict(self):
        pack = RubricPack(
            pack_id="test",
            domain="testing",
            description="Test",
            rubrics=[{"id": "test/r1", "type": "q"}],
            version="1.0.0",
            author="a",
        )
        d = pack.as_dict()
        assert d["pack_id"] == "test"
        assert d["domain"] == "testing"
        assert len(d["rubrics"]) == 1
        assert d["version"] == "1.0.0"
        assert d["author"] == "a"

    def test_from_dict(self):
        data = {
            "pack_id": "custom",
            "domain": "custom_domain",
            "description": "A custom pack",
            "rubrics": [{"id": "custom/r1"}],
            "version": "2.0.0",
            "author": "me",
        }
        pack = RubricPack.from_dict(data)
        assert pack.pack_id == "custom"
        assert pack.version == "2.0.0"
        assert len(pack.rubrics) == 1

    def test_from_dict_defaults(self):
        data = {"pack_id": "x", "domain": "d"}
        pack = RubricPack.from_dict(data)
        assert pack.description == ""
        assert pack.rubrics == []
        assert pack.version == "0.0.0"
        assert pack.author == ""

    def test_roundtrip(self):
        pack = RubricPack(
            pack_id="rt",
            domain="roundtrip",
            description="Round trip test",
            rubrics=[{"id": "rt/r1", "type": "safety", "description": "Test rubric"}],
            version="1.2.3",
            author="evalyn",
        )
        pack2 = RubricPack.from_dict(pack.as_dict())
        assert pack2.pack_id == pack.pack_id
        assert pack2.domain == pack.domain
        assert pack2.rubrics == pack.rubrics

    def test_as_dict_copies_rubrics(self):
        rubrics = [{"id": "x/r1"}]
        pack = RubricPack("x", "d", "desc", rubrics, "1.0", "a")
        d = pack.as_dict()
        d["rubrics"][0]["id"] = "changed"
        assert pack.rubrics[0]["id"] == "x/r1"


# ---------------------------------------------------------------------------
# PackRegistry
# ---------------------------------------------------------------------------


class TestPackRegistry:
    def _make_pack(self, pack_id: str, domain: str = "test") -> RubricPack:
        return RubricPack(
            pack_id=pack_id,
            domain=domain,
            description=f"Pack {pack_id} for {domain}",
            rubrics=[
                {"id": f"{pack_id}/r1", "type": "quality", "description": "Rubric 1"},
                {"id": f"{pack_id}/r2", "type": "safety", "description": "Rubric 2"},
            ],
            version="1.0.0",
            author="tester",
        )

    def test_register_and_get(self):
        reg = PackRegistry()
        pack = self._make_pack("alpha")
        reg.register(pack)
        assert reg.get("alpha") is pack

    def test_get_missing(self):
        reg = PackRegistry()
        assert reg.get("nonexistent") is None

    def test_list_packs(self):
        reg = PackRegistry()
        p1 = self._make_pack("a")
        p2 = self._make_pack("b")
        reg.register(p1)
        reg.register(p2)
        packs = reg.list_packs()
        assert len(packs) == 2
        assert p1 in packs
        assert p2 in packs

    def test_list_packs_empty(self):
        reg = PackRegistry()
        assert reg.list_packs() == []

    def test_list_domains(self):
        reg = PackRegistry()
        reg.register(self._make_pack("a", domain="medical"))
        reg.register(self._make_pack("b", domain="legal"))
        reg.register(self._make_pack("c", domain="medical"))
        domains = reg.list_domains()
        assert domains == ["legal", "medical"]

    def test_list_domains_empty(self):
        reg = PackRegistry()
        assert reg.list_domains() == []

    def test_search_by_pack_id(self):
        reg = PackRegistry()
        reg.register(self._make_pack("medical", domain="healthcare"))
        reg.register(self._make_pack("legal", domain="law"))
        results = reg.search("medical")
        assert len(results) == 1
        assert results[0].pack_id == "medical"

    def test_search_by_domain(self):
        reg = PackRegistry()
        reg.register(self._make_pack("pack1", domain="healthcare"))
        reg.register(self._make_pack("pack2", domain="law"))
        results = reg.search("healthcare")
        assert len(results) == 1
        assert results[0].domain == "healthcare"

    def test_search_by_description(self):
        reg = PackRegistry()
        reg.register(self._make_pack("pack1", domain="d1"))
        results = reg.search("Pack pack1")
        assert len(results) == 1

    def test_search_case_insensitive(self):
        reg = PackRegistry()
        reg.register(self._make_pack("Medical", domain="Healthcare"))
        results = reg.search("medical")
        assert len(results) == 1

    def test_search_no_match(self):
        reg = PackRegistry()
        reg.register(self._make_pack("alpha"))
        results = reg.search("zzz_nonexistent")
        assert results == []

    def test_install(self):
        reg = PackRegistry()
        pack = self._make_pack("alpha")
        reg.register(pack)
        target: list = []
        count = reg.install(pack, target)
        assert count == 2
        assert len(target) == 2
        assert target[0]["id"] == "alpha/r1"
        assert target[1]["id"] == "alpha/r2"

    def test_install_appends(self):
        reg = PackRegistry()
        pack = self._make_pack("alpha")
        reg.register(pack)
        target: list = [{"id": "existing/r0"}]
        reg.install(pack, target)
        assert len(target) == 3
        assert target[0]["id"] == "existing/r0"

    def test_uninstall(self):
        reg = PackRegistry()
        pack = self._make_pack("alpha")
        reg.register(pack)
        target: list = []
        reg.install(pack, target)
        assert len(target) == 2
        count = reg.uninstall("alpha", target)
        assert count == 2
        assert len(target) == 0

    def test_uninstall_preserves_other(self):
        reg = PackRegistry()
        pack_a = self._make_pack("alpha")
        pack_b = self._make_pack("beta")
        reg.register(pack_a)
        reg.register(pack_b)
        target: list = []
        reg.install(pack_a, target)
        reg.install(pack_b, target)
        assert len(target) == 4
        reg.uninstall("alpha", target)
        assert len(target) == 2
        assert all(r["id"].startswith("beta/") for r in target)

    def test_uninstall_nothing_to_remove(self):
        reg = PackRegistry()
        target: list = [{"id": "other/r1"}]
        count = reg.uninstall("nonexistent", target)
        assert count == 0
        assert len(target) == 1

    def test_register_overwrites(self):
        reg = PackRegistry()
        p1 = self._make_pack("alpha")
        p2 = RubricPack("alpha", "new_domain", "New desc", [], "2.0.0", "new_author")
        reg.register(p1)
        reg.register(p2)
        assert reg.get("alpha") is p2
        assert len(reg.list_packs()) == 1


# ---------------------------------------------------------------------------
# BUILTIN_PACKS
# ---------------------------------------------------------------------------


class TestBuiltinPacks:
    def test_three_packs(self):
        packs = BUILTIN_PACKS.list_packs()
        assert len(packs) == 3

    def test_pack_ids(self):
        ids = {p.pack_id for p in BUILTIN_PACKS.list_packs()}
        assert ids == {"medical", "legal", "finance"}

    def test_domains(self):
        domains = BUILTIN_PACKS.list_domains()
        assert len(domains) == 3
        assert "healthcare" in domains
        assert "legal" in domains
        assert "finance" in domains

    def test_medical_pack_rubrics(self):
        pack = BUILTIN_PACKS.get("medical")
        assert pack is not None
        assert len(pack.rubrics) == 4
        ids = [r["id"] for r in pack.rubrics]
        assert "medical/hipaa_compliance" in ids
        assert "medical/clinical_accuracy" in ids
        assert "medical/patient_safety" in ids
        assert "medical/drug_interaction" in ids

    def test_legal_pack_rubrics(self):
        pack = BUILTIN_PACKS.get("legal")
        assert pack is not None
        assert len(pack.rubrics) == 4
        ids = [r["id"] for r in pack.rubrics]
        assert "legal/jurisdictional_accuracy" in ids
        assert "legal/precedent_citation" in ids
        assert "legal/privilege_preservation" in ids
        assert "legal/legal_reasoning" in ids

    def test_finance_pack_rubrics(self):
        pack = BUILTIN_PACKS.get("finance")
        assert pack is not None
        assert len(pack.rubrics) == 4
        ids = [r["id"] for r in pack.rubrics]
        assert "finance/sec_compliance" in ids
        assert "finance/fiduciary_duty" in ids
        assert "finance/risk_disclosure" in ids
        assert "finance/financial_accuracy" in ids

    def test_all_rubrics_have_five_levels(self):
        for pack in BUILTIN_PACKS.list_packs():
            for rubric in pack.rubrics:
                assert "rubric" in rubric, f"Rubric {rubric['id']} missing 'rubric' key"
                levels = rubric["rubric"]
                assert len(levels) == 5, f"Rubric {rubric['id']} has {len(levels)} levels, expected 5"
                for score in ["5", "4", "3", "2", "1"]:
                    assert score in levels, f"Rubric {rubric['id']} missing level {score}"

    def test_all_rubrics_have_required_fields(self):
        required_keys = {"id", "type", "description", "category", "scope", "prompt", "rubric"}
        for pack in BUILTIN_PACKS.list_packs():
            for rubric in pack.rubrics:
                missing = required_keys - set(rubric.keys())
                assert not missing, f"Rubric {rubric.get('id', '?')} missing keys: {missing}"

    def test_all_rubrics_have_prompt_text(self):
        for pack in BUILTIN_PACKS.list_packs():
            for rubric in pack.rubrics:
                assert len(rubric["prompt"]) > 20, f"Rubric {rubric['id']} has very short prompt"

    def test_pack_versions(self):
        for pack in BUILTIN_PACKS.list_packs():
            assert pack.version == "1.0.0"

    def test_pack_authors(self):
        for pack in BUILTIN_PACKS.list_packs():
            assert pack.author == "evalyn"

    def test_search_medical(self):
        results = BUILTIN_PACKS.search("medical")
        assert len(results) >= 1
        assert any(p.pack_id == "medical" for p in results)

    def test_search_compliance(self):
        results = BUILTIN_PACKS.search("compliance")
        # Medical and finance packs mention compliance in their descriptions
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# format_pack_listing
# ---------------------------------------------------------------------------


class TestFormatPackListing:
    def test_empty_registry(self):
        reg = PackRegistry()
        assert format_pack_listing(reg) == "No packs registered."

    def test_has_all_pack_ids(self):
        listing = format_pack_listing(BUILTIN_PACKS)
        for pack_id in ["medical", "legal", "finance"]:
            assert pack_id in listing

    def test_has_versions(self):
        listing = format_pack_listing(BUILTIN_PACKS)
        assert "v1.0.0" in listing

    def test_has_rubric_counts(self):
        listing = format_pack_listing(BUILTIN_PACKS)
        assert "4 rubrics" in listing

    def test_single_pack(self):
        reg = PackRegistry()
        reg.register(RubricPack("test", "d", "Test pack", [], "0.1.0", "me"))
        listing = format_pack_listing(reg)
        assert "test" in listing
        assert "v0.1.0" in listing


# ---------------------------------------------------------------------------
# format_pack_detail
# ---------------------------------------------------------------------------


class TestFormatPackDetail:
    def test_medical_detail(self):
        pack = BUILTIN_PACKS.get("medical")
        assert pack is not None
        detail = format_pack_detail(pack)
        assert "Pack: medical" in detail
        assert "Domain: healthcare" in detail
        assert "Author: evalyn" in detail
        assert "medical/hipaa_compliance" in detail

    def test_has_rubric_types(self):
        pack = BUILTIN_PACKS.get("legal")
        assert pack is not None
        detail = format_pack_detail(pack)
        assert "Type:" in detail

    def test_has_categories(self):
        pack = BUILTIN_PACKS.get("finance")
        assert pack is not None
        detail = format_pack_detail(pack)
        assert "Category:" in detail

    def test_has_descriptions(self):
        pack = BUILTIN_PACKS.get("medical")
        assert pack is not None
        detail = format_pack_detail(pack)
        assert "Description:" in detail

    def test_rubric_count_in_header(self):
        pack = BUILTIN_PACKS.get("legal")
        assert pack is not None
        detail = format_pack_detail(pack)
        assert "Rubrics (4)" in detail
