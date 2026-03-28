"""Tests for calibration curriculum module."""

from evalyn_sdk.calibration.curriculum import (
    CurriculumPlan,
    CurriculumStage,
    DifficultyScore,
    create_curriculum,
    estimate_difficulty,
    get_stage_items,
    sort_by_difficulty,
)


# ---------------------------------------------------------------------------
# estimate_difficulty
# ---------------------------------------------------------------------------


def test_estimate_difficulty_high_score_is_easy():
    d = estimate_difficulty("item1", score=0.9)
    assert d.difficulty < 0.2
    assert d.item_id == "item1"


def test_estimate_difficulty_low_score_is_hard():
    d = estimate_difficulty("item2", score=0.1)
    assert d.difficulty > 0.8


def test_estimate_difficulty_perfect_score():
    d = estimate_difficulty("item3", score=1.0)
    assert d.difficulty == 0.0


def test_estimate_difficulty_zero_score():
    d = estimate_difficulty("item4", score=0.0)
    assert d.difficulty == 1.0


def test_estimate_difficulty_low_confidence_increases():
    high_conf = estimate_difficulty("a", score=0.5, confidence=1.0)
    low_conf = estimate_difficulty("b", score=0.5, confidence=0.3)
    assert low_conf.difficulty > high_conf.difficulty


def test_estimate_difficulty_low_confidence_adds_02():
    d = estimate_difficulty("x", score=0.5, confidence=0.4)
    # base difficulty = 0.5, + 0.2 = 0.7
    assert abs(d.difficulty - 0.7) < 1e-9


def test_estimate_difficulty_low_confidence_clamps_to_1():
    d = estimate_difficulty("y", score=0.0, confidence=0.1)
    # base = 1.0, + 0.2 = 1.2, clamped to 1.0
    assert d.difficulty == 1.0


def test_estimate_difficulty_reason_includes_score():
    d = estimate_difficulty("z", score=0.6)
    assert "score=0.6" in d.reason


def test_estimate_difficulty_reason_includes_confidence():
    d = estimate_difficulty("z", score=0.6, confidence=0.3)
    assert "confidence" in d.reason


# ---------------------------------------------------------------------------
# sort_by_difficulty
# ---------------------------------------------------------------------------


def test_sort_by_difficulty_ascending():
    items = [
        DifficultyScore("hard", 0.9),
        DifficultyScore("easy", 0.1),
        DifficultyScore("mid", 0.5),
    ]
    result = sort_by_difficulty(items)
    assert [d.item_id for d in result] == ["easy", "mid", "hard"]


def test_sort_by_difficulty_empty():
    assert sort_by_difficulty([]) == []


def test_sort_by_difficulty_single():
    items = [DifficultyScore("only", 0.5)]
    result = sort_by_difficulty(items)
    assert len(result) == 1
    assert result[0].item_id == "only"


# ---------------------------------------------------------------------------
# create_curriculum
# ---------------------------------------------------------------------------


def test_create_curriculum_correct_stages():
    items = [estimate_difficulty(f"i{i}", score=i / 5) for i in range(6)]
    plan = create_curriculum(items, num_stages=3)
    assert plan.num_stages == 3
    assert plan.total_items == 6
    assert len(plan.stages) == 3


def test_create_curriculum_stage_order():
    items = [estimate_difficulty(f"i{i}", score=i / 5) for i in range(6)]
    plan = create_curriculum(items, num_stages=3)
    # Stage 1 should have easiest items (highest scores -> lowest difficulty)
    for s in plan.stages:
        assert s.stage >= 1
    # Difficulty ranges should be ascending across stages
    for i in range(len(plan.stages) - 1):
        assert plan.stages[i].difficulty_range[1] <= plan.stages[i + 1].difficulty_range[0] or \
               plan.stages[i].difficulty_range[1] <= plan.stages[i + 1].difficulty_range[1]


def test_create_curriculum_single_stage():
    items = [DifficultyScore(f"i{i}", i * 0.25) for i in range(4)]
    plan = create_curriculum(items, num_stages=1)
    assert plan.num_stages == 1
    assert plan.total_items == 4
    assert plan.stages[0].num_items == 4


def test_create_curriculum_empty_items():
    plan = create_curriculum([], num_stages=3)
    assert plan.total_items == 0
    assert plan.num_stages == 0
    assert plan.stages == []


def test_create_curriculum_more_stages_than_items():
    items = [DifficultyScore("a", 0.3), DifficultyScore("b", 0.7)]
    plan = create_curriculum(items, num_stages=5)
    # Should clamp to 2 stages
    assert plan.num_stages == 2
    assert plan.total_items == 2


def test_create_curriculum_single_item():
    items = [DifficultyScore("only", 0.5)]
    plan = create_curriculum(items, num_stages=3)
    assert plan.num_stages == 1
    assert plan.total_items == 1


# ---------------------------------------------------------------------------
# get_stage_items
# ---------------------------------------------------------------------------


def test_get_stage_items_cumulative():
    items = [DifficultyScore(f"i{i}", i * 0.1) for i in range(6)]
    plan = create_curriculum(items, num_stages=3)
    stage1 = get_stage_items(plan, 1)
    stage2 = get_stage_items(plan, 2)
    stage3 = get_stage_items(plan, 3)
    # Each subsequent stage includes all previous items
    assert len(stage1) < len(stage2) < len(stage3)
    assert len(stage3) == 6
    # Stage 2 includes stage 1 items
    for item_id in stage1:
        assert item_id in stage2


def test_get_stage_items_stage_1():
    items = [DifficultyScore(f"i{i}", i * 0.2) for i in range(5)]
    plan = create_curriculum(items, num_stages=5)
    stage1 = get_stage_items(plan, 1)
    assert len(stage1) == 1


def test_get_stage_items_empty_plan():
    plan = CurriculumPlan(stages=[], total_items=0, num_stages=0)
    assert get_stage_items(plan, 1) == []


def test_get_stage_items_clamps_high_stage():
    items = [DifficultyScore(f"i{i}", i * 0.25) for i in range(4)]
    plan = create_curriculum(items, num_stages=2)
    # Stage 10 should clamp to max stage
    result = get_stage_items(plan, 10)
    assert len(result) == 4


# ---------------------------------------------------------------------------
# CurriculumPlan format_text
# ---------------------------------------------------------------------------


def test_curriculum_plan_format_text():
    plan = CurriculumPlan(
        stages=[
            CurriculumStage(1, ["a", "b"], (0.0, 0.3), 2),
            CurriculumStage(2, ["c", "d"], (0.4, 0.7), 2),
        ],
        total_items=4,
        num_stages=2,
    )
    text = plan.format_text()
    assert "2 stages" in text
    assert "4 items" in text
    assert "Stage 1" in text
    assert "Stage 2" in text


# ---------------------------------------------------------------------------
# as_dict / from_dict roundtrips
# ---------------------------------------------------------------------------


def test_difficulty_score_as_dict():
    d = DifficultyScore("item1", 0.5, "test reason")
    data = d.as_dict()
    assert data["item_id"] == "item1"
    assert data["difficulty"] == 0.5
    assert data["reason"] == "test reason"


def test_curriculum_stage_as_dict():
    s = CurriculumStage(1, ["a", "b"], (0.0, 0.5), 2)
    data = s.as_dict()
    assert data["stage"] == 1
    assert data["item_ids"] == ["a", "b"]
    assert data["difficulty_range"] == [0.0, 0.5]


def test_curriculum_plan_roundtrip():
    plan = CurriculumPlan(
        stages=[
            CurriculumStage(1, ["a"], (0.0, 0.3), 1),
            CurriculumStage(2, ["b", "c"], (0.4, 0.9), 2),
        ],
        total_items=3,
        num_stages=2,
    )
    data = plan.as_dict()
    restored = CurriculumPlan.from_dict(data)
    assert restored.total_items == plan.total_items
    assert restored.num_stages == plan.num_stages
    assert len(restored.stages) == len(plan.stages)
    assert restored.stages[0].item_ids == ["a"]
    assert restored.stages[1].difficulty_range == (0.4, 0.9)


def test_curriculum_plan_from_dict_empty():
    data = {"stages": [], "total_items": 0, "num_stages": 0}
    plan = CurriculumPlan.from_dict(data)
    assert plan.stages == []
    assert plan.total_items == 0
