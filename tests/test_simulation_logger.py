"""Tests for simulation_logger.py — covers dataclasses, template tracking,
winner attribution, diff, margin, and JSONL serialization."""

import json
import os
import tempfile
import numpy as np
import pytest

from simulation_logger import (
    InputStats,
    AreaActivation,
    StepLog,
    AssemblyTemplate,
    SimulationLogger,
)


# ── InputStats ──────────────────────────────────────────────────────────────

class TestInputStats:
    def test_from_array_basic(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        s = InputStats.from_array(arr)
        assert s.min == 1.0
        assert s.max == 5.0
        assert s.mean == pytest.approx(3.0)
        assert s.median == pytest.approx(3.0)
        assert s.std == pytest.approx(np.std(arr))
        assert s.p25 == pytest.approx(np.percentile(arr, 25))
        assert s.p75 == pytest.approx(np.percentile(arr, 75))

    def test_from_array_empty(self):
        s = InputStats.from_array(np.array([]))
        assert s.min == 0.0
        assert s.max == 0.0
        assert s.mean == 0.0

    def test_from_array_single(self):
        s = InputStats.from_array(np.array([42.0]))
        assert s.min == 42.0
        assert s.max == 42.0
        assert s.mean == 42.0
        assert s.std == 0.0

    def test_from_none(self):
        s = InputStats.from_array(None)
        assert s.min == 0.0


# ── Template registration & overlap ────────────────────────────────────────

class TestTemplateOverlap:
    def test_register_template(self):
        logger = SimulationLogger(log_dir=tempfile.mkdtemp())
        logger.register_template("q0", area="q0", neuron_ids=range(100))
        assert "q0" in logger.templates
        assert len(logger.templates["q0"].neuron_ids) == 100

    def test_perfect_overlap(self):
        logger = SimulationLogger(log_dir=tempfile.mkdtemp())
        logger.register_template("q0", area="q0", neuron_ids=range(100))
        act = logger.record_activation(
            area="q0",
            winners=list(range(100)),
        )
        assert act.template_name == "q0"
        assert act.template_overlap == pytest.approx(1.0)

    def test_partial_overlap(self):
        logger = SimulationLogger(log_dir=tempfile.mkdtemp())
        logger.register_template("q0", area="q0", neuron_ids=range(100))
        # 50 matching + 50 non-matching → Jaccard = 50 / 150
        winners = list(range(50)) + list(range(100, 150))
        act = logger.record_activation(area="q0", winners=winners)
        assert act.template_name == "q0"
        assert act.template_overlap == pytest.approx(50 / 150)

    def test_no_template_for_area(self):
        logger = SimulationLogger(log_dir=tempfile.mkdtemp())
        logger.register_template("q0", area="q0", neuron_ids=range(100))
        act = logger.record_activation(area="OTHER", winners=[1, 2, 3])
        assert act.template_name is None
        assert act.template_overlap == 0.0

    def test_best_template_selected(self):
        """When multiple templates match the same area, pick the one with
        highest Jaccard."""
        logger = SimulationLogger(log_dir=tempfile.mkdtemp())
        logger.register_template("good", area="A", neuron_ids=range(100))
        logger.register_template("bad", area="A", neuron_ids=range(500, 600))
        act = logger.record_activation(area="A", winners=list(range(100)))
        assert act.template_name == "good"
        assert act.template_overlap == pytest.approx(1.0)


# ── Winner diff (gained / lost / overlap_with_prev) ────────────────────────

class TestWinnerDiff:
    def test_first_step_all_gained(self):
        logger = SimulationLogger(log_dir=tempfile.mkdtemp())
        act = logger.record_activation(area="A", winners=[0, 1, 2])
        assert act.winners_gained == [0, 1, 2]
        assert act.winners_lost == []
        assert act.overlap_with_prev == 0.0

    def test_no_change(self):
        logger = SimulationLogger(log_dir=tempfile.mkdtemp())
        logger.record_activation(area="A", winners=[0, 1, 2])
        act2 = logger.record_activation(area="A", winners=[0, 1, 2])
        assert act2.winners_gained == []
        assert act2.winners_lost == []
        assert act2.overlap_with_prev == pytest.approx(1.0)

    def test_partial_change(self):
        logger = SimulationLogger(log_dir=tempfile.mkdtemp())
        logger.record_activation(area="A", winners=[0, 1, 2])
        act2 = logger.record_activation(area="A", winners=[1, 2, 3])
        assert act2.winners_gained == [3]
        assert act2.winners_lost == [0]
        # Jaccard = 2 / 4 = 0.5
        assert act2.overlap_with_prev == pytest.approx(0.5)

    def test_independent_areas(self):
        """Diffs are tracked per-area independently."""
        logger = SimulationLogger(log_dir=tempfile.mkdtemp())
        logger.record_activation(area="A", winners=[0, 1])
        logger.record_activation(area="B", winners=[10, 11])
        act_a = logger.record_activation(area="A", winners=[0, 2])
        act_b = logger.record_activation(area="B", winners=[10, 12])
        assert act_a.winners_gained == [2]
        assert act_a.winners_lost == [1]
        assert act_b.winners_gained == [12]
        assert act_b.winners_lost == [11]


# ── Activation margin ──────────────────────────────────────────────────────

class TestActivationMargin:
    def test_margin_computed(self):
        logger = SimulationLogger(log_dir=tempfile.mkdtemp())
        # k=3 winners → margin = sorted[2] - sorted[3]
        all_inputs = np.array([10.0, 5.0, 8.0, 3.0, 1.0])
        act = logger.record_activation(
            area="A",
            winners=[0, 2, 1],   # top-3: 10, 8, 5
            all_inputs=all_inputs,
        )
        # sorted desc: 10, 8, 5, 3, 1 → margin = 5 - 3 = 2
        assert act.activation_margin == pytest.approx(2.0)

    def test_margin_zero_when_no_all_inputs(self):
        logger = SimulationLogger(log_dir=tempfile.mkdtemp())
        act = logger.record_activation(area="A", winners=[0, 1])
        assert act.activation_margin == 0.0


# ── Attribution ────────────────────────────────────────────────────────────

class TestAttribution:
    def test_attribution_stored(self):
        logger = SimulationLogger(log_dir=tempfile.mkdtemp())
        attr = {"stim_a": 500.0, "area_B": 300.0}
        act = logger.record_activation(
            area="A", winners=[0, 1, 2], attribution=attr
        )
        assert act.attribution == attr

    def test_attribution_default_empty(self):
        logger = SimulationLogger(log_dir=tempfile.mkdtemp())
        act = logger.record_activation(area="A", winners=[0])
        assert act.attribution == {}


# ── StepLog ────────────────────────────────────────────────────────────────

class TestStepLog:
    def test_log_step_increments(self):
        logger = SimulationLogger(log_dir=tempfile.mkdtemp())
        act = logger.record_activation(area="A", winners=[0])
        s1 = logger.log_step(activations={"A": act})
        s2 = logger.log_step(activations={"A": act})
        assert s1.step == 0
        assert s2.step == 1
        assert len(logger.steps) == 2

    def test_dfa_fields(self):
        logger = SimulationLogger(log_dir=tempfile.mkdtemp())
        act = logger.record_activation(area="q1", winners=[0, 1])
        s = logger.log_step(
            activations={"q1": act},
            dfa_state="q1",
            dfa_symbol="b",
            dfa_transition="(q0, a) → q1",
        )
        assert s.dfa_state == "q1"
        assert s.dfa_symbol == "b"
        assert s.dfa_transition == "(q0, a) → q1"


# ── JSONL save / load round-trip ───────────────────────────────────────────

class TestSerialization:
    def test_save_and_load(self):
        tmpdir = tempfile.mkdtemp()
        logger = SimulationLogger(log_dir=tmpdir)
        logger.register_template("q0", area="q0", neuron_ids=range(5))

        act = logger.record_activation(
            area="q0",
            winners=[0, 1, 2, 3, 4],
            winner_inputs=[10.0, 9.0, 8.0, 7.0, 6.0],
            all_inputs=np.arange(20, dtype=np.float64),
            num_new_winners=0,
            attribution={"stim_a": 50.0},
        )
        logger.log_step(
            activations={"q0": act},
            dfa_state="q0",
            dfa_symbol="a",
        )
        logger.save()

        # Load and verify
        rows = logger.load()
        assert len(rows) == 1
        row = rows[0]
        assert row["step"] == 0
        assert row["dfa_state"] == "q0"
        assert row["dfa_symbol"] == "a"

        q0_act = row["activations"]["q0"]
        assert q0_act["winners"] == [0, 1, 2, 3, 4]
        assert q0_act["template_name"] == "q0"
        assert q0_act["template_overlap"] == pytest.approx(1.0)
        assert q0_act["attribution"]["stim_a"] == 50.0

    def test_multiple_steps(self):
        tmpdir = tempfile.mkdtemp()
        logger = SimulationLogger(log_dir=tmpdir)
        for i in range(10):
            act = logger.record_activation(area="A", winners=[i])
            logger.log_step(activations={"A": act})
        logger.save()
        rows = logger.load()
        assert len(rows) == 10
        assert [r["step"] for r in rows] == list(range(10))

    def test_file_is_valid_jsonl(self):
        tmpdir = tempfile.mkdtemp()
        logger = SimulationLogger(log_dir=tmpdir)
        act = logger.record_activation(area="A", winners=[1, 2])
        logger.log_step(activations={"A": act})
        logger.save()

        with open(logger.log_path, "r") as f:
            lines = f.readlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert "activations" in parsed


# ── Integration: end-to-end with Brain ─────────────────────────────────────

class TestIntegrationWithBrain:
    """Ensure the logger can work alongside actual Brain objects."""

    def test_record_brain_winners(self):
        """Record winners from a Brain area (simulated)."""
        logger = SimulationLogger(log_dir=tempfile.mkdtemp())
        logger.register_template("assembly_A", area="A", neuron_ids=range(10))

        # Simulate: winners = [0..9], inputs across 100 neurons
        all_inputs = np.zeros(100)
        all_inputs[:10] = np.arange(10, 20)  # winners get 10-19
        winners = list(range(10))
        winner_inputs = all_inputs[:10].tolist()

        act = logger.record_activation(
            area="A",
            winners=winners,
            winner_inputs=winner_inputs,
            all_inputs=all_inputs,
            attribution={"stim_X": 100.0},
        )
        assert act.template_overlap == pytest.approx(1.0)
        assert act.activation_margin == pytest.approx(10.0)  # 10 - 0
        assert act.input_stats is not None
        assert act.input_stats.max == pytest.approx(19.0)
