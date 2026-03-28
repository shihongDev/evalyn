"""Tests for annotation_flywheel module."""

from __future__ import annotations

from evalyn_sdk.calibration.annotation_flywheel import (
    AnnotationItem,
    FlywheelConfig,
    FlywheelState,
    compute_agreement_rate,
    create_annotation_queue,
    record_human_annotation,
    select_next_batch,
    should_continue,
    update_flywheel_state,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _aitem(
    item_id: str = "i1",
    text: str = "hello",
    ai_label: str = "good",
    **kwargs,
) -> AnnotationItem:
    return AnnotationItem(item_id=item_id, text=text, ai_label=ai_label, **kwargs)


# ---------------------------------------------------------------------------
# AnnotationItem as_dict / from_dict
# ---------------------------------------------------------------------------


class TestAnnotationItemRoundtrip:
    def test_roundtrip_defaults(self):
        item = _aitem()
        restored = AnnotationItem.from_dict(item.as_dict())
        assert restored.item_id == "i1"
        assert restored.text == "hello"
        assert restored.ai_label == "good"
        assert restored.annotated is False

    def test_roundtrip_annotated(self):
        item = _aitem(human_label="bad", agreement=False, annotated=True)
        restored = AnnotationItem.from_dict(item.as_dict())
        assert restored.human_label == "bad"
        assert restored.agreement is False
        assert restored.annotated is True

    def test_as_dict_keys(self):
        d = _aitem().as_dict()
        expected = {"item_id", "text", "ai_label", "human_label", "agreement", "annotated"}
        assert set(d.keys()) == expected


# ---------------------------------------------------------------------------
# FlywheelConfig
# ---------------------------------------------------------------------------


class TestFlywheelConfig:
    def test_defaults(self):
        cfg = FlywheelConfig()
        assert cfg.batch_size == 10
        assert cfg.target_agreement == 0.9
        assert cfg.max_iterations == 10

    def test_from_dict(self):
        cfg = FlywheelConfig.from_dict({
            "batch_size": 5,
            "target_agreement": 0.95,
            "max_iterations": 20,
        })
        assert cfg.batch_size == 5
        assert cfg.target_agreement == 0.95

    def test_roundtrip(self):
        cfg = FlywheelConfig(batch_size=3, target_agreement=0.8, max_iterations=5)
        restored = FlywheelConfig.from_dict(cfg.as_dict())
        assert restored.batch_size == cfg.batch_size
        assert restored.target_agreement == cfg.target_agreement
        assert restored.max_iterations == cfg.max_iterations


# ---------------------------------------------------------------------------
# FlywheelState
# ---------------------------------------------------------------------------


class TestFlywheelState:
    def test_format_text(self):
        state = FlywheelState(
            queue=[_aitem()],
            annotated_count=1,
            agreement_rate=0.75,
            improvement_per_batch=[0.25, 0.10],
        )
        text = state.format_text()
        assert "Flywheel State" in text
        assert "Queue size: 1" in text
        assert "Agreement rate: 0.7500" in text
        assert "Batches completed: 2" in text
        assert "Batch 1:" in text

    def test_roundtrip(self):
        state = FlywheelState(
            queue=[_aitem("a"), _aitem("b")],
            annotated_count=1,
            agreement_rate=0.5,
            improvement_per_batch=[0.5],
        )
        restored = FlywheelState.from_dict(state.as_dict())
        assert len(restored.queue) == 2
        assert restored.annotated_count == 1
        assert restored.agreement_rate == 0.5
        assert restored.improvement_per_batch == [0.5]

    def test_empty_state(self):
        state = FlywheelState()
        assert state.annotated_count == 0
        assert state.agreement_rate == 0.0
        text = state.format_text()
        assert "Queue size: 0" in text


# ---------------------------------------------------------------------------
# create_annotation_queue
# ---------------------------------------------------------------------------


class TestCreateAnnotationQueue:
    def test_basic(self):
        items = [
            {"id": "a", "text": "foo"},
            {"id": "b", "text": "bar"},
        ]
        labels = {"a": "good", "b": "bad"}
        queue = create_annotation_queue(items, labels)
        assert len(queue) == 2
        assert queue[0].item_id == "a"
        assert queue[0].ai_label == "good"
        assert queue[1].ai_label == "bad"

    def test_missing_label(self):
        items = [{"id": "x", "text": "hello"}]
        queue = create_annotation_queue(items, {})
        assert queue[0].ai_label == ""

    def test_empty(self):
        queue = create_annotation_queue([], {})
        assert queue == []


# ---------------------------------------------------------------------------
# record_human_annotation
# ---------------------------------------------------------------------------


class TestRecordHumanAnnotation:
    def test_agreement(self):
        item = _aitem(ai_label="good")
        result = record_human_annotation(item, "good")
        assert result.annotated is True
        assert result.agreement is True
        assert result.human_label == "good"

    def test_disagreement(self):
        item = _aitem(ai_label="good")
        result = record_human_annotation(item, "bad")
        assert result.annotated is True
        assert result.agreement is False
        assert result.human_label == "bad"

    def test_preserves_fields(self):
        item = _aitem("x", "some text", "label_a")
        result = record_human_annotation(item, "label_b")
        assert result.item_id == "x"
        assert result.text == "some text"
        assert result.ai_label == "label_a"


# ---------------------------------------------------------------------------
# compute_agreement_rate
# ---------------------------------------------------------------------------


class TestComputeAgreementRate:
    def test_full_agreement(self):
        items = [
            _aitem("a", annotated=True, agreement=True),
            _aitem("b", annotated=True, agreement=True),
        ]
        assert compute_agreement_rate(items) == 1.0

    def test_no_agreement(self):
        items = [
            _aitem("a", annotated=True, agreement=False),
            _aitem("b", annotated=True, agreement=False),
        ]
        assert compute_agreement_rate(items) == 0.0

    def test_partial_agreement(self):
        items = [
            _aitem("a", annotated=True, agreement=True),
            _aitem("b", annotated=True, agreement=False),
        ]
        assert compute_agreement_rate(items) == 0.5

    def test_ignores_unannotated(self):
        items = [
            _aitem("a", annotated=True, agreement=True),
            _aitem("b", annotated=False),
        ]
        assert compute_agreement_rate(items) == 1.0

    def test_empty(self):
        assert compute_agreement_rate([]) == 0.0

    def test_all_unannotated(self):
        items = [_aitem("a"), _aitem("b")]
        assert compute_agreement_rate(items) == 0.0


# ---------------------------------------------------------------------------
# select_next_batch
# ---------------------------------------------------------------------------


class TestSelectNextBatch:
    def test_selects_unannotated(self):
        items = [
            _aitem("a", annotated=True),
            _aitem("b"),
            _aitem("c"),
        ]
        batch = select_next_batch(items, batch_size=2)
        assert len(batch) == 2
        assert batch[0].item_id == "b"
        assert batch[1].item_id == "c"

    def test_respects_batch_size(self):
        items = [_aitem(f"i{i}") for i in range(20)]
        batch = select_next_batch(items, batch_size=5)
        assert len(batch) == 5

    def test_all_annotated(self):
        items = [_aitem("a", annotated=True), _aitem("b", annotated=True)]
        batch = select_next_batch(items, batch_size=10)
        assert batch == []

    def test_empty_queue(self):
        assert select_next_batch([], batch_size=5) == []


# ---------------------------------------------------------------------------
# should_continue
# ---------------------------------------------------------------------------


class TestShouldContinue:
    def test_below_target(self):
        state = FlywheelState(agreement_rate=0.5, improvement_per_batch=[0.5])
        cfg = FlywheelConfig(target_agreement=0.9, max_iterations=10)
        assert should_continue(state, cfg) is True

    def test_above_target(self):
        state = FlywheelState(agreement_rate=0.95, improvement_per_batch=[0.5])
        cfg = FlywheelConfig(target_agreement=0.9, max_iterations=10)
        assert should_continue(state, cfg) is False

    def test_at_target(self):
        state = FlywheelState(agreement_rate=0.9, improvement_per_batch=[0.5])
        cfg = FlywheelConfig(target_agreement=0.9, max_iterations=10)
        assert should_continue(state, cfg) is False

    def test_max_iterations_reached(self):
        state = FlywheelState(
            agreement_rate=0.5,
            improvement_per_batch=[0.1] * 10,
        )
        cfg = FlywheelConfig(target_agreement=0.9, max_iterations=10)
        assert should_continue(state, cfg) is False

    def test_zero_iterations(self):
        state = FlywheelState(agreement_rate=0.0)
        cfg = FlywheelConfig(target_agreement=0.9, max_iterations=10)
        assert should_continue(state, cfg) is True


# ---------------------------------------------------------------------------
# update_flywheel_state
# ---------------------------------------------------------------------------


class TestUpdateFlywheelState:
    def test_updates_items(self):
        queue = [_aitem("a"), _aitem("b")]
        state = FlywheelState(queue=queue)
        annotated_a = record_human_annotation(queue[0], "good")
        new_state = update_flywheel_state(state, [annotated_a])
        assert new_state.queue[0].annotated is True
        assert new_state.queue[1].annotated is False
        assert new_state.annotated_count == 1

    def test_tracks_improvement(self):
        queue = [_aitem("a"), _aitem("b")]
        state = FlywheelState(queue=queue, agreement_rate=0.0)
        annotated_a = record_human_annotation(queue[0], "good")
        new_state = update_flywheel_state(state, [annotated_a])
        # 1 annotated, 1 agrees -> rate=1.0, improvement=1.0
        assert new_state.agreement_rate == 1.0
        assert len(new_state.improvement_per_batch) == 1
        assert new_state.improvement_per_batch[0] == 1.0

    def test_preserves_existing_batches(self):
        queue = [_aitem("a"), _aitem("b")]
        state = FlywheelState(
            queue=queue,
            agreement_rate=0.5,
            improvement_per_batch=[0.5],
        )
        annotated_b = record_human_annotation(queue[1], "good")
        new_state = update_flywheel_state(state, [annotated_b])
        assert len(new_state.improvement_per_batch) == 2
        assert new_state.improvement_per_batch[0] == 0.5

    def test_empty_newly_annotated(self):
        queue = [_aitem("a")]
        state = FlywheelState(queue=queue, agreement_rate=0.0)
        new_state = update_flywheel_state(state, [])
        assert new_state.agreement_rate == 0.0
        assert len(new_state.improvement_per_batch) == 1
