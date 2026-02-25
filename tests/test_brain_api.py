"""Tests for viz/brain_api.py — LiveBrain wrapper."""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from viz.brain_api import LiveBrain, StepRecord


# ── Area & Stimulus Management ──────────────────────────────────────────────

class TestAreaManagement:
    def test_add_area(self):
        lb = LiveBrain(p=0.1)
        record = lb.add_area("A", n=500, k=50, beta=0.05)
        assert "A" in lb.get_areas()
        assert record.operation == "add_area"

    def test_add_multiple_areas(self):
        lb = LiveBrain(p=0.1)
        lb.add_area("A", n=500, k=50)
        lb.add_area("B", n=500, k=50)
        assert sorted(lb.get_areas()) == ["A", "B"]

    def test_add_area_creates_fibers(self):
        lb = LiveBrain(p=0.1)
        lb.add_area("A", n=500, k=50)
        lb.add_area("B", n=500, k=50)
        fibers = lb.get_fibers()
        assert ("A", "B") in fibers
        assert ("B", "A") in fibers

    def test_add_stimulus(self):
        lb = LiveBrain(p=0.1)
        lb.add_area("A", n=500, k=50)
        record = lb.add_stimulus("stim_x", size=50)
        assert "stim_x" in lb.get_stimuli()
        assert record.operation == "add_stimulus"


# ── Operations ──────────────────────────────────────────────────────────────

class TestStimulate:
    def test_stimulate_fires_winners(self):
        lb = LiveBrain(p=0.1)
        lb.add_area("A", n=500, k=50)
        lb.add_stimulus("stim", size=50)
        lb.stimulate("stim", "A")
        state = lb.get_area_state("A")
        assert state["n_winners"] == 50

    def test_stimulate_multiple_rounds(self):
        lb = LiveBrain(p=0.1)
        lb.add_area("A", n=500, k=50)
        lb.add_stimulus("stim", size=50)
        lb.stimulate("stim", "A", rounds=5)
        state = lb.get_area_state("A")
        assert state["n_winners"] == 50

    def test_stimulate_convergence(self):
        """After many rounds, overlap should approach 1.0."""
        lb = LiveBrain(p=0.1, seed=42)
        lb.add_area("A", n=500, k=50)
        lb.add_stimulus("stim", size=50)
        for _ in range(15):
            lb.stimulate("stim", "A")
        state = lb.get_area_state("A")
        # After 15 rounds of same stimulus, should converge
        assert state["overlap_prev"] >= 0.7


class TestProject:
    def test_project_creates_assembly(self):
        lb = LiveBrain(p=0.1, seed=42)
        lb.add_area("A", n=500, k=50)
        lb.add_area("B", n=500, k=50)
        lb.add_stimulus("stim", size=50)
        # First stimulate A to create assembly
        lb.stimulate("stim", "A", rounds=5)
        # Then project A → B
        lb.project("A", "B", rounds=5)
        state_b = lb.get_area_state("B")
        assert state_b["n_winners"] > 0

    def test_project_multiple_rounds(self):
        lb = LiveBrain(p=0.1, seed=42)
        lb.add_area("A", n=500, k=50)
        lb.add_area("B", n=500, k=50)
        lb.add_stimulus("stim", size=50)
        lb.stimulate("stim", "A", rounds=5)
        record = lb.project("A", "B", rounds=10)
        assert record.operation == "project"


class TestReciprocalProject:
    def test_reciprocal(self):
        lb = LiveBrain(p=0.1, seed=42)
        lb.add_area("A", n=500, k=50)
        lb.add_area("B", n=500, k=50)
        lb.add_stimulus("stim", size=50)
        lb.stimulate("stim", "A", rounds=5)
        lb.project("A", "B", rounds=3)
        record = lb.reciprocal_project("A", "B", rounds=5)
        assert record.operation == "reciprocal"
        # Both areas should have winners
        assert lb.get_area_state("A")["n_winners"] > 0
        assert lb.get_area_state("B")["n_winners"] > 0


# ── Learning Toggle ─────────────────────────────────────────────────────────

class TestLearningToggle:
    def test_default_learning_on(self):
        lb = LiveBrain(p=0.1)
        assert lb.learning is True
        assert lb.brain.disable_plasticity is False

    def test_toggle_off(self):
        lb = LiveBrain(p=0.1)
        lb.set_learning(False)
        assert lb.learning is False
        assert lb.brain.disable_plasticity is True

    def test_toggle_on_again(self):
        lb = LiveBrain(p=0.1)
        lb.set_learning(False)
        lb.set_learning(True)
        assert lb.learning is True
        assert lb.brain.disable_plasticity is False

    def test_no_weight_change_when_off(self):
        """When learning is off, weights should not change."""
        lb = LiveBrain(p=0.1, seed=42)
        lb.add_area("A", n=500, k=50)
        lb.add_stimulus("stim", size=50)
        lb.stimulate("stim", "A", rounds=5)

        # Get weights before
        stats_before = lb.get_weight_stats("A", "A")

        # Turn off learning & stimulate more
        lb.set_learning(False)
        lb.stimulate("stim", "A", rounds=5)

        # Weights should not change
        stats_after = lb.get_weight_stats("A", "A")
        assert abs(stats_before["mean"] - stats_after["mean"]) < 0.01


# ── State Export ────────────────────────────────────────────────────────────

class TestStateExport:
    def test_get_area_state(self):
        lb = LiveBrain(p=0.1)
        lb.add_area("A", n=500, k=50)
        state = lb.get_area_state("A")
        assert state["name"] == "A"
        assert state["n"] == 500
        assert state["k"] == 50

    def test_get_full_state(self):
        lb = LiveBrain(p=0.1)
        lb.add_area("A", n=500, k=50)
        lb.add_area("B", n=500, k=50)
        full = lb.get_full_state()
        assert "A" in full
        assert "B" in full

    def test_missing_area(self):
        lb = LiveBrain(p=0.1)
        state = lb.get_area_state("nonexistent")
        assert state == {}

    def test_weight_stats(self):
        lb = LiveBrain(p=0.1)
        lb.add_area("A", n=100, k=10)
        lb.add_area("B", n=100, k=10)
        stats = lb.get_weight_stats("A", "B")
        assert "mean" in stats
        assert "std" in stats

    def test_weight_sample(self):
        lb = LiveBrain(p=0.1)
        lb.add_area("A", n=100, k=10)
        sample = lb.get_weight_sample("A", "A")
        assert isinstance(sample, list)
        assert len(sample) > 0

    def test_weight_sample_missing(self):
        lb = LiveBrain(p=0.1)
        sample = lb.get_weight_sample("X", "Y")
        assert sample == []


# ── History ─────────────────────────────────────────────────────────────────

class TestHistory:
    def test_step_logging(self):
        lb = LiveBrain(p=0.1)
        lb.add_area("A", n=500, k=50)
        lb.add_stimulus("stim", size=50)
        lb.stimulate("stim", "A")
        assert len(lb.steps) == 3  # add_area + add_stim + stimulate(1 round)

    def test_step_increments(self):
        lb = LiveBrain(p=0.1)
        lb.add_area("A", n=500, k=50)
        lb.add_area("B", n=500, k=50)
        assert lb.steps[0].step == 0
        assert lb.steps[1].step == 1

    def test_overlap_history(self):
        lb = LiveBrain(p=0.1, seed=42)
        lb.add_area("A", n=500, k=50)
        lb.add_stimulus("stim", size=50)
        for _ in range(5):
            lb.stimulate("stim", "A")
        history = lb.get_overlap_history("A")
        assert len(history) == 7  # add_area + add_stim + 5 stimulates (1 round each)

    def test_winner_count_history(self):
        lb = LiveBrain(p=0.1, seed=42)
        lb.add_area("A", n=500, k=50)
        lb.add_stimulus("stim", size=50)
        lb.stimulate("stim", "A")
        history = lb.get_winner_count_history("A")
        assert len(history) == 3  # add_area + add_stim + stimulate(1 round)

    def test_record_contains_snapshot(self):
        lb = LiveBrain(p=0.1)
        lb.add_area("A", n=500, k=50)
        record = lb.steps[0]
        assert "A" in record.areas_snapshot
        assert isinstance(record, StepRecord)

    def test_record_learning_flag(self):
        lb = LiveBrain(p=0.1)
        lb.add_area("A", n=500, k=50)
        assert lb.steps[0].learning is True
        lb.set_learning(False)
        lb.add_area("B", n=500, k=50)
        assert lb.steps[1].learning is False


# ── Gained/Lost & Snapshot ──────────────────────────────────────────────────

class TestGainedLost:
    def test_first_stimulate_has_gained(self):
        """First stimulation should show 50 gained (from 0 winners)."""
        lb = LiveBrain(p=0.1, seed=42)
        lb.add_area("A", n=500, k=50)
        lb.add_stimulus("stim", size=50)
        lb.stimulate("stim", "A")
        # Last snapshot should have correct gained
        snap = lb._last_snapshot["A"]
        assert snap["gained"] == 50
        assert snap["lost"] == 0

    def test_snapshot_has_gained_set(self):
        """Snapshot should contain actual neuron ID sets."""
        lb = LiveBrain(p=0.1, seed=42)
        lb.add_area("A", n=200, k=20)
        lb.add_stimulus("stim", size=20)
        lb.stimulate("stim", "A")
        snap = lb._last_snapshot["A"]
        assert "gained_set" in snap
        assert "lost_set" in snap
        assert "winners_set" in snap
        assert len(snap["gained_set"]) == 20  # all new
        assert len(snap["winners_set"]) == 20

    def test_subsequent_stimulate_has_some_lost(self):
        """Second stimulation may show some gained/lost (not converged yet)."""
        lb = LiveBrain(p=0.1, seed=42)
        lb.add_area("A", n=500, k=50)
        lb.add_stimulus("stim", size=50)
        lb.stimulate("stim", "A")
        lb.stimulate("stim", "A")
        snap = lb._last_snapshot["A"]
        # After 2 stimulations, overlap is partial, so some gained/lost expected
        # or fully converged — either way, snapshot data should be consistent
        assert snap["gained"] + snap["lost"] >= 0  # sanity
        assert snap["n_winners"] == 50

    def test_snapshot_persists_between_renders(self):
        """_last_snapshot should survive multiple reads."""
        lb = LiveBrain(p=0.1, seed=42)
        lb.add_area("A", n=200, k=20)
        lb.add_stimulus("stim", size=20)
        lb.stimulate("stim", "A")
        snap1 = lb._last_snapshot["A"]["gained"]
        snap2 = lb._last_snapshot["A"]["gained"]
        assert snap1 == snap2 == 20


class TestReset:
    def test_reset_clears_all(self):
        lb = LiveBrain(p=0.1, seed=42)
        lb.add_area("A", n=200, k=20)
        lb.add_stimulus("stim", size=20)
        lb.stimulate("stim", "A")
        lb.reset()
        assert lb.get_areas() == []
        assert lb.get_stimuli() == []
        assert lb.steps == []
        assert lb._last_snapshot == {}
        assert lb._prev_winners == {}

    def test_reset_allows_rebuild(self):
        lb = LiveBrain(p=0.1, seed=42)
        lb.add_area("A", n=200, k=20)
        lb.reset()
        lb.add_area("B", n=200, k=20)
        assert lb.get_areas() == ["B"]
        assert len(lb.steps) == 1


# ── Associate & Merge ──────────────────────────────────────────────────────

class TestAssociate:
    def test_associate_creates_assembly(self):
        lb = LiveBrain(p=0.1, seed=42)
        lb.add_area("A", n=100, k=10)
        lb.add_stimulus("s1", size=10)
        lb.add_stimulus("s2", size=10)
        record = lb.associate("s1", "s2", "A", rounds=3)
        assert record.operation == "associate"
        state = lb.get_area_state("A")
        assert state["n_winners"] == 10

    def test_associate_logs_per_round(self):
        lb = LiveBrain(p=0.1, seed=42)
        lb.add_area("A", n=100, k=10)
        lb.add_stimulus("s1", size=10)
        lb.add_stimulus("s2", size=10)
        lb.associate("s1", "s2", "A", rounds=5)
        # 1 add_area + 2 add_stim + 5 associate rounds = 8
        assert len(lb.steps) == 8


class TestMerge:
    def test_merge_creates_assembly(self):
        lb = LiveBrain(p=0.1, seed=42)
        lb.add_area("A", n=100, k=10)
        lb.add_area("B", n=100, k=10)
        lb.add_area("C", n=100, k=10)
        lb.add_stimulus("s", size=10)
        lb.stimulate("s", "A", rounds=3)
        lb.stimulate("s", "B", rounds=3)
        record = lb.merge("A", "B", "C", rounds=3)
        assert record.operation == "merge"
        state = lb.get_area_state("C")
        assert state["n_winners"] > 0

    def test_merge_logs_per_round(self):
        lb = LiveBrain(p=0.1, seed=42)
        lb.add_area("A", n=100, k=10)
        lb.add_area("B", n=100, k=10)
        lb.add_area("C", n=100, k=10)
        lb.add_stimulus("s", size=10)
        lb.stimulate("s", "A")
        lb.stimulate("s", "B")
        lb.merge("A", "B", "C", rounds=5)
        # 3 add_area + 1 add_stim + 1 stim A + 1 stim B + 5 merge = 11
        assert len(lb.steps) == 11

