from __future__ import annotations

import json

import pytest

from evalyn_sdk.calibration.checkpoint import (
    CheckpointConfig,
    CheckpointData,
    CheckpointManager,
)


# -- CheckpointData tests --


def test_checkpoint_data_as_dict():
    cp = CheckpointData(metric_id="m1", optimizer="basic", step=3, best_score=0.8)
    d = cp.as_dict()
    assert d["metric_id"] == "m1"
    assert d["step"] == 3
    assert d["best_score"] == 0.8


def test_checkpoint_data_from_dict():
    d = {"metric_id": "m1", "optimizer": "opro", "step": 5, "best_score": 0.9}
    cp = CheckpointData.from_dict(d)
    assert cp.metric_id == "m1"
    assert cp.optimizer == "opro"
    assert cp.step == 5
    assert cp.best_score == 0.9


def test_checkpoint_data_roundtrip():
    cp = CheckpointData(
        metric_id="m1",
        optimizer="ape",
        step=10,
        best_score=0.75,
        best_prompt="best",
        current_prompt="current",
        scores_history=[0.5, 0.6, 0.75],
        metadata={"key": "val"},
        timestamp="2026-01-01T00:00:00+00:00",
    )
    restored = CheckpointData.from_dict(cp.as_dict())
    assert restored.as_dict() == cp.as_dict()


def test_checkpoint_data_defaults():
    cp = CheckpointData(metric_id="m1", optimizer="basic", step=0)
    assert cp.best_score == 0.0
    assert cp.best_prompt == ""
    assert cp.current_prompt == ""
    assert cp.scores_history == []
    assert cp.metadata == {}
    assert cp.timestamp == ""


# -- CheckpointConfig tests --


def test_checkpoint_config_defaults():
    cfg = CheckpointConfig()
    assert cfg.enabled is True
    assert cfg.save_every_n_steps == 5
    assert cfg.keep_last_n == 3


def test_checkpoint_config_as_dict():
    cfg = CheckpointConfig(enabled=False, save_every_n_steps=10, keep_last_n=5)
    d = cfg.as_dict()
    assert d == {"enabled": False, "save_every_n_steps": 10, "keep_last_n": 5}


def test_checkpoint_config_from_dict():
    cfg = CheckpointConfig.from_dict({"enabled": False, "save_every_n_steps": 2, "keep_last_n": 1})
    assert cfg.enabled is False
    assert cfg.save_every_n_steps == 2
    assert cfg.keep_last_n == 1


def test_checkpoint_config_from_dict_defaults():
    cfg = CheckpointConfig.from_dict({})
    assert cfg.enabled is True
    assert cfg.save_every_n_steps == 5
    assert cfg.keep_last_n == 3


# -- CheckpointManager: save/load --


def test_save_and_load_latest():
    mgr = CheckpointManager(CheckpointConfig())
    cp1 = CheckpointData(metric_id="m1", optimizer="basic", step=5)
    cp2 = CheckpointData(metric_id="m1", optimizer="basic", step=10)
    mgr.save(cp1)
    mgr.save(cp2)
    latest = mgr.load_latest("m1")
    assert latest is not None
    assert latest.step == 10


def test_load_latest_no_checkpoints():
    mgr = CheckpointManager(CheckpointConfig())
    assert mgr.load_latest("nonexistent") is None


def test_load_best():
    mgr = CheckpointManager(CheckpointConfig())
    mgr.save(CheckpointData(metric_id="m1", optimizer="basic", step=5, best_score=0.5))
    mgr.save(CheckpointData(metric_id="m1", optimizer="basic", step=10, best_score=0.9))
    mgr.save(CheckpointData(metric_id="m1", optimizer="basic", step=15, best_score=0.7))
    best = mgr.load_best("m1")
    assert best is not None
    assert best.best_score == 0.9


def test_load_best_no_checkpoints():
    mgr = CheckpointManager(CheckpointConfig())
    assert mgr.load_best("nonexistent") is None


# -- should_checkpoint --


def test_should_checkpoint_at_boundary():
    mgr = CheckpointManager(CheckpointConfig(save_every_n_steps=5))
    assert mgr.should_checkpoint(0) is True
    assert mgr.should_checkpoint(5) is True
    assert mgr.should_checkpoint(10) is True


def test_should_checkpoint_not_at_boundary():
    mgr = CheckpointManager(CheckpointConfig(save_every_n_steps=5))
    assert mgr.should_checkpoint(1) is False
    assert mgr.should_checkpoint(3) is False
    assert mgr.should_checkpoint(7) is False


def test_should_checkpoint_every_step():
    mgr = CheckpointManager(CheckpointConfig(save_every_n_steps=1))
    assert mgr.should_checkpoint(0) is True
    assert mgr.should_checkpoint(1) is True
    assert mgr.should_checkpoint(99) is True


# -- keep_last_n trimming --


def test_keep_last_n_trims_old():
    mgr = CheckpointManager(CheckpointConfig(keep_last_n=2))
    for i in range(5):
        mgr.save(CheckpointData(metric_id="m1", optimizer="basic", step=i))
    cps = mgr.list_checkpoints("m1")
    assert len(cps) == 2
    assert cps[0].step == 3
    assert cps[1].step == 4


def test_keep_last_n_one():
    mgr = CheckpointManager(CheckpointConfig(keep_last_n=1))
    mgr.save(CheckpointData(metric_id="m1", optimizer="basic", step=1))
    mgr.save(CheckpointData(metric_id="m1", optimizer="basic", step=2))
    cps = mgr.list_checkpoints("m1")
    assert len(cps) == 1
    assert cps[0].step == 2


# -- clear --


def test_clear_removes_metric():
    mgr = CheckpointManager(CheckpointConfig())
    mgr.save(CheckpointData(metric_id="m1", optimizer="basic", step=5))
    mgr.save(CheckpointData(metric_id="m2", optimizer="basic", step=5))
    mgr.clear("m1")
    assert mgr.list_checkpoints("m1") == []
    assert len(mgr.list_checkpoints("m2")) == 1


def test_clear_nonexistent_metric():
    mgr = CheckpointManager(CheckpointConfig())
    mgr.clear("nonexistent")  # should not raise


def test_clear_all():
    mgr = CheckpointManager(CheckpointConfig())
    mgr.save(CheckpointData(metric_id="m1", optimizer="basic", step=5))
    mgr.save(CheckpointData(metric_id="m2", optimizer="basic", step=10))
    mgr.clear_all()
    assert mgr.list_checkpoints("m1") == []
    assert mgr.list_checkpoints("m2") == []


# -- list_checkpoints --


def test_list_checkpoints():
    mgr = CheckpointManager(CheckpointConfig())
    mgr.save(CheckpointData(metric_id="m1", optimizer="basic", step=1))
    mgr.save(CheckpointData(metric_id="m1", optimizer="basic", step=2))
    cps = mgr.list_checkpoints("m1")
    assert len(cps) == 2


def test_list_checkpoints_empty():
    mgr = CheckpointManager(CheckpointConfig())
    assert mgr.list_checkpoints("m1") == []


# -- as_dict / from_dict --


def test_manager_as_dict_from_dict():
    mgr = CheckpointManager(CheckpointConfig(save_every_n_steps=3, keep_last_n=2))
    mgr.save(CheckpointData(metric_id="m1", optimizer="opro", step=3, best_score=0.8))
    mgr.save(CheckpointData(metric_id="m1", optimizer="opro", step=6, best_score=0.9))
    d = mgr.as_dict()
    restored = CheckpointManager.from_dict(d)
    assert restored.config.save_every_n_steps == 3
    assert len(restored.list_checkpoints("m1")) == 2


# -- to_json / from_json --


def test_to_json_from_json_roundtrip():
    mgr = CheckpointManager(CheckpointConfig())
    mgr.save(CheckpointData(metric_id="m1", optimizer="basic", step=5, best_score=0.7))
    mgr.save(CheckpointData(metric_id="m2", optimizer="ape", step=10, best_score=0.85))
    json_str = mgr.to_json()
    restored = CheckpointManager.from_json(json_str)
    assert restored.load_latest("m1").step == 5
    assert restored.load_latest("m2").best_score == 0.85


def test_to_json_produces_valid_json():
    mgr = CheckpointManager(CheckpointConfig())
    mgr.save(CheckpointData(metric_id="m1", optimizer="basic", step=1))
    parsed = json.loads(mgr.to_json())
    assert "config" in parsed
    assert "checkpoints" in parsed


# -- Edge cases --


def test_save_sets_timestamp_when_empty():
    mgr = CheckpointManager(CheckpointConfig())
    cp = CheckpointData(metric_id="m1", optimizer="basic", step=1, timestamp="")
    mgr.save(cp)
    assert cp.timestamp != ""


def test_save_preserves_existing_timestamp():
    mgr = CheckpointManager(CheckpointConfig())
    ts = "2026-01-01T00:00:00+00:00"
    cp = CheckpointData(metric_id="m1", optimizer="basic", step=1, timestamp=ts)
    mgr.save(cp)
    assert cp.timestamp == ts


def test_single_step_checkpoint():
    mgr = CheckpointManager(CheckpointConfig(keep_last_n=10))
    cp = CheckpointData(metric_id="m1", optimizer="basic", step=0, best_score=0.1)
    mgr.save(cp)
    assert mgr.load_latest("m1").step == 0
    assert mgr.load_best("m1").best_score == 0.1
