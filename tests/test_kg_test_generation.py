"""Tests for kg_test_generation module."""

from __future__ import annotations

import pytest

from evalyn_sdk.kg_test_generation import (
    GeneratedQuestion,
    KGEdge,
    KGNode,
    KnowledgeGraph,
    build_knowledge_graph,
    extract_entities,
    extract_relations,
    format_question_set,
    generate_factual_questions,
    generate_reasoning_questions,
)


# ---------------------------------------------------------------------------
# KGNode tests
# ---------------------------------------------------------------------------


class TestKGNode:
    def test_basic_creation(self):
        node = KGNode(node_id="n1", label="Test Node", node_type="entity")
        assert node.node_id == "n1"
        assert node.label == "Test Node"
        assert node.node_type == "entity"
        assert node.properties == {}

    def test_with_properties(self):
        node = KGNode(
            node_id="n2",
            label="Foo",
            node_type="concept",
            properties={"color": "blue"},
        )
        assert node.properties == {"color": "blue"}

    def test_as_dict(self):
        node = KGNode(
            node_id="n1", label="X", node_type="fact", properties={"k": "v"}
        )
        d = node.as_dict()
        assert d == {
            "node_id": "n1",
            "label": "X",
            "node_type": "fact",
            "properties": {"k": "v"},
        }

    def test_from_dict(self):
        d = {"node_id": "n1", "label": "X", "node_type": "entity"}
        node = KGNode.from_dict(d)
        assert node.node_id == "n1"
        assert node.properties == {}

    def test_roundtrip(self):
        node = KGNode(
            node_id="abc", label="Round", node_type="concept", properties={"a": "b"}
        )
        assert KGNode.from_dict(node.as_dict()) == node


# ---------------------------------------------------------------------------
# KGEdge tests
# ---------------------------------------------------------------------------


class TestKGEdge:
    def test_basic_creation(self):
        edge = KGEdge(source="a", target="b", relation="knows")
        assert edge.weight == 1.0

    def test_custom_weight(self):
        edge = KGEdge(source="a", target="b", relation="rel", weight=0.5)
        assert edge.weight == 0.5

    def test_as_dict(self):
        edge = KGEdge(source="a", target="b", relation="r", weight=2.0)
        d = edge.as_dict()
        assert d == {"source": "a", "target": "b", "relation": "r", "weight": 2.0}

    def test_from_dict(self):
        d = {"source": "x", "target": "y", "relation": "z"}
        edge = KGEdge.from_dict(d)
        assert edge.weight == 1.0

    def test_roundtrip(self):
        edge = KGEdge(source="s", target="t", relation="r", weight=0.7)
        assert KGEdge.from_dict(edge.as_dict()) == edge


# ---------------------------------------------------------------------------
# KnowledgeGraph tests
# ---------------------------------------------------------------------------


class TestKnowledgeGraph:
    def test_empty_graph(self):
        g = KnowledgeGraph()
        assert g.nodes == []
        assert g.edges == []

    def test_as_dict(self):
        g = KnowledgeGraph(
            nodes=[KGNode(node_id="n1", label="A", node_type="entity")],
            edges=[KGEdge(source="n1", target="n2", relation="r")],
        )
        d = g.as_dict()
        assert len(d["nodes"]) == 1
        assert len(d["edges"]) == 1

    def test_from_dict_empty(self):
        g = KnowledgeGraph.from_dict({})
        assert g.nodes == []
        assert g.edges == []

    def test_roundtrip(self):
        g = KnowledgeGraph(
            nodes=[
                KGNode(node_id="n1", label="A", node_type="entity"),
                KGNode(node_id="n2", label="B", node_type="concept"),
            ],
            edges=[KGEdge(source="n1", target="n2", relation="links")],
        )
        g2 = KnowledgeGraph.from_dict(g.as_dict())
        assert len(g2.nodes) == 2
        assert len(g2.edges) == 1
        assert g2.nodes[0].label == "A"


# ---------------------------------------------------------------------------
# GeneratedQuestion tests
# ---------------------------------------------------------------------------


class TestGeneratedQuestion:
    def test_defaults(self):
        q = GeneratedQuestion(question="Q?", expected_answer="A")
        assert q.difficulty == "medium"
        assert q.question_type == "factual"
        assert q.source_nodes == []

    def test_as_dict(self):
        q = GeneratedQuestion(
            question="Q?",
            expected_answer="A",
            source_nodes=["n1"],
            difficulty="hard",
            question_type="reasoning",
        )
        d = q.as_dict()
        assert d["difficulty"] == "hard"
        assert d["source_nodes"] == ["n1"]

    def test_from_dict_defaults(self):
        q = GeneratedQuestion.from_dict({"question": "Q?", "expected_answer": "A"})
        assert q.difficulty == "medium"

    def test_roundtrip(self):
        q = GeneratedQuestion(
            question="Q?",
            expected_answer="A",
            source_nodes=["a", "b"],
            difficulty="easy",
            question_type="factual",
        )
        assert GeneratedQuestion.from_dict(q.as_dict()) == q


# ---------------------------------------------------------------------------
# extract_entities tests
# ---------------------------------------------------------------------------


class TestExtractEntities:
    def test_capitalized_phrase(self):
        nodes = extract_entities("The United States is large.")
        labels = {n.label for n in nodes}
        assert "United States" in labels

    def test_quoted_term(self):
        nodes = extract_entities('The concept of "machine learning" is important.')
        labels = {n.label for n in nodes}
        assert "machine learning" in labels

    def test_quoted_term_is_concept(self):
        nodes = extract_entities('We use "gradient descent" here.')
        concepts = [n for n in nodes if n.node_type == "concept"]
        assert len(concepts) >= 1

    def test_is_a_pattern(self):
        nodes = extract_entities("A dog is a domestic animal.")
        labels = {n.label for n in nodes}
        assert "domestic animal" in labels

    def test_are_pattern(self):
        nodes = extract_entities("Cats are small mammals.")
        labels = {n.label for n in nodes}
        assert "small mammals" in labels

    def test_empty_text(self):
        assert extract_entities("") == []

    def test_no_entities(self):
        assert extract_entities("just a plain lowercase sentence.") == []

    def test_deduplicate_by_id(self):
        text = "New York is big. We love New York."
        nodes = extract_entities(text)
        ids = [n.node_id for n in nodes]
        assert len(ids) == len(set(ids))

    def test_multiple_types(self):
        text = 'John Smith said "hello world" is an example phrase.'
        nodes = extract_entities(text)
        types = {n.node_type for n in nodes}
        assert len(types) >= 2

    def test_node_id_deterministic(self):
        nodes1 = extract_entities("Big Data is important.")
        nodes2 = extract_entities("Big Data is important.")
        assert [n.node_id for n in nodes1] == [n.node_id for n in nodes2]


# ---------------------------------------------------------------------------
# extract_relations tests
# ---------------------------------------------------------------------------


class TestExtractRelations:
    def test_basic_relation(self):
        text = "John Smith uses Big Data."
        entities = [
            KGNode(node_id="john_smith", label="John Smith", node_type="entity"),
            KGNode(node_id="big_data", label="Big Data", node_type="entity"),
        ]
        edges = extract_relations(text, entities)
        assert len(edges) >= 1
        assert edges[0].source == "john_smith"
        assert edges[0].target == "big_data"

    def test_no_matching_entities(self):
        text = "nothing relevant here"
        entities = [
            KGNode(node_id="x", label="Foo Bar", node_type="entity"),
        ]
        edges = extract_relations(text, entities)
        assert edges == []

    def test_deduplication(self):
        text = "Alpha Beta uses Gamma Delta. Alpha Beta uses Gamma Delta again."
        entities = [
            KGNode(node_id="alpha_beta", label="Alpha Beta", node_type="entity"),
            KGNode(node_id="gamma_delta", label="Gamma Delta", node_type="entity"),
        ]
        edges = extract_relations(text, entities)
        keys = [(e.source, e.target, e.relation) for e in edges]
        assert len(keys) == len(set(keys))

    def test_self_loop_excluded(self):
        text = "Some Thing uses Some Thing."
        entities = [
            KGNode(node_id="some_thing", label="Some Thing", node_type="entity"),
        ]
        edges = extract_relations(text, entities)
        assert edges == []


# ---------------------------------------------------------------------------
# build_knowledge_graph tests
# ---------------------------------------------------------------------------


class TestBuildKnowledgeGraph:
    def test_empty_texts(self):
        g = build_knowledge_graph([])
        assert g.nodes == []
        assert g.edges == []

    def test_single_text(self):
        g = build_knowledge_graph(["The United Nations supports Human Rights."])
        assert len(g.nodes) >= 1

    def test_multiple_texts_dedup_nodes(self):
        texts = [
            "Big Data is a concept.",
            "Big Data is important.",
        ]
        g = build_knowledge_graph(texts)
        ids = [n.node_id for n in g.nodes]
        # big_data should appear only once
        assert ids.count("big_data") <= 1

    def test_edges_from_texts(self):
        texts = [
            "Alpha Beta uses Gamma Delta.",
            "Alpha Beta creates Epsilon Zeta.",
        ]
        g = build_knowledge_graph(texts)
        assert len(g.edges) >= 1


# ---------------------------------------------------------------------------
# generate_factual_questions tests
# ---------------------------------------------------------------------------


class TestGenerateFactualQuestions:
    def _sample_graph(self) -> KnowledgeGraph:
        return KnowledgeGraph(
            nodes=[
                KGNode(node_id="a", label="Alpha", node_type="entity"),
                KGNode(node_id="b", label="Beta", node_type="concept"),
                KGNode(node_id="c", label="Gamma", node_type="fact"),
            ],
            edges=[
                KGEdge(source="a", target="b", relation="uses"),
                KGEdge(source="b", target="c", relation="creates"),
            ],
        )

    def test_generates_questions(self):
        qs = generate_factual_questions(self._sample_graph(), n=5)
        assert len(qs) <= 5
        assert len(qs) >= 1

    def test_node_based_question(self):
        qs = generate_factual_questions(self._sample_graph(), n=1)
        assert "What is Alpha?" == qs[0].question

    def test_edge_based_question_included(self):
        qs = generate_factual_questions(self._sample_graph(), n=5)
        rel_qs = [q for q in qs if "relationship" in q.question]
        assert len(rel_qs) >= 1

    def test_empty_graph(self):
        qs = generate_factual_questions(KnowledgeGraph(), n=5)
        assert qs == []

    def test_n_limit_respected(self):
        g = self._sample_graph()
        qs = generate_factual_questions(g, n=2)
        assert len(qs) == 2

    def test_question_type_is_factual(self):
        qs = generate_factual_questions(self._sample_graph(), n=3)
        assert all(q.question_type == "factual" for q in qs)

    def test_source_nodes_populated(self):
        qs = generate_factual_questions(self._sample_graph(), n=1)
        assert len(qs[0].source_nodes) >= 1


# ---------------------------------------------------------------------------
# generate_reasoning_questions tests
# ---------------------------------------------------------------------------


class TestGenerateReasoningQuestions:
    def _chain_graph(self) -> KnowledgeGraph:
        return KnowledgeGraph(
            nodes=[
                KGNode(node_id="a", label="Alpha", node_type="entity"),
                KGNode(node_id="b", label="Beta", node_type="entity"),
                KGNode(node_id="c", label="Gamma", node_type="entity"),
            ],
            edges=[
                KGEdge(source="a", target="b", relation="uses"),
                KGEdge(source="b", target="c", relation="creates"),
            ],
        )

    def test_generates_reasoning(self):
        qs = generate_reasoning_questions(self._chain_graph(), n=3)
        assert len(qs) >= 1

    def test_multi_hop_content(self):
        qs = generate_reasoning_questions(self._chain_graph(), n=1)
        q = qs[0]
        assert "Alpha" in q.question
        assert "Gamma" in q.question
        assert "Beta" in q.question

    def test_difficulty_is_hard(self):
        qs = generate_reasoning_questions(self._chain_graph(), n=1)
        assert qs[0].difficulty == "hard"

    def test_type_is_reasoning(self):
        qs = generate_reasoning_questions(self._chain_graph(), n=1)
        assert qs[0].question_type == "reasoning"

    def test_source_nodes_three(self):
        qs = generate_reasoning_questions(self._chain_graph(), n=1)
        assert len(qs[0].source_nodes) == 3

    def test_no_cycle_back(self):
        # A->B->A should not produce a question
        g = KnowledgeGraph(
            nodes=[
                KGNode(node_id="a", label="A", node_type="entity"),
                KGNode(node_id="b", label="B", node_type="entity"),
            ],
            edges=[
                KGEdge(source="a", target="b", relation="uses"),
                KGEdge(source="b", target="a", relation="creates"),
            ],
        )
        qs = generate_reasoning_questions(g, n=5)
        assert qs == []

    def test_empty_graph(self):
        qs = generate_reasoning_questions(KnowledgeGraph(), n=3)
        assert qs == []

    def test_n_limit(self):
        g = KnowledgeGraph(
            nodes=[
                KGNode(node_id="a", label="A", node_type="entity"),
                KGNode(node_id="b", label="B", node_type="entity"),
                KGNode(node_id="c", label="C", node_type="entity"),
                KGNode(node_id="d", label="D", node_type="entity"),
            ],
            edges=[
                KGEdge(source="a", target="b", relation="r1"),
                KGEdge(source="b", target="c", relation="r2"),
                KGEdge(source="b", target="d", relation="r3"),
            ],
        )
        qs = generate_reasoning_questions(g, n=1)
        assert len(qs) == 1


# ---------------------------------------------------------------------------
# format_question_set tests
# ---------------------------------------------------------------------------


class TestFormatQuestionSet:
    def test_empty(self):
        assert format_question_set([]) == "No questions generated."

    def test_single_question(self):
        qs = [GeneratedQuestion(question="Q?", expected_answer="A.")]
        out = format_question_set(qs)
        assert "Q1:" in out
        assert "A1:" in out
        assert "Q?" in out

    def test_multiple_questions(self):
        qs = [
            GeneratedQuestion(question="Q1?", expected_answer="A1."),
            GeneratedQuestion(question="Q2?", expected_answer="A2."),
        ]
        out = format_question_set(qs)
        assert "Q1:" in out
        assert "Q2:" in out

    def test_includes_metadata(self):
        qs = [
            GeneratedQuestion(
                question="Q?",
                expected_answer="A.",
                difficulty="hard",
                question_type="reasoning",
            )
        ]
        out = format_question_set(qs)
        assert "difficulty=hard" in out
        assert "type=reasoning" in out
