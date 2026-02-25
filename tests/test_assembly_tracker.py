"""Tests for viz/assembly_tracker.py — covers ID assignment, convergence,
drift, loss, process_log, and queries."""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from viz.assembly_tracker import AssemblyTracker, TrackedAssembly


class TestIDAssignment:
    def test_first_update_creates_assembly(self):
        tracker = AssemblyTracker()
        asm_id = tracker.update("A", list(range(100)), step=0)
        assert asm_id == "A:0"
        assert len(tracker.assemblies) == 1

    def test_second_area_gets_own_id(self):
        tracker = AssemblyTracker()
        id_a = tracker.update("A", list(range(100)), step=0)
        id_b = tracker.update("B", list(range(100)), step=0)
        assert id_a == "A:0"
        assert id_b == "B:0"

    def test_same_area_same_assembly(self):
        tracker = AssemblyTracker()
        id1 = tracker.update("A", list(range(100)), step=0)
        # Same winners → same assembly
        id2 = tracker.update("A", list(range(100)), step=1)
        assert id1 == id2

    def test_empty_winners_returns_none(self):
        tracker = AssemblyTracker()
        asm_id = tracker.update("A", [], step=0)
        assert asm_id is None


class TestConvergence:
    def test_becomes_stable_after_n_steps(self):
        tracker = AssemblyTracker(convergence_steps=3)
        winners = list(range(100))
        for i in range(5):
            tracker.update("A", winners, step=i)
        asm = tracker.assemblies["A:0"]
        assert asm.status == "stable"
        assert asm.consecutive_stable >= 3

    def test_not_stable_if_too_few_steps(self):
        tracker = AssemblyTracker(convergence_steps=10)
        winners = list(range(100))
        for i in range(3):
            tracker.update("A", winners, step=i)
        asm = tracker.assemblies["A:0"]
        assert asm.status != "stable"


class TestDrift:
    def test_drift_detected(self):
        tracker = AssemblyTracker(
            convergence_threshold=0.8,
            convergence_steps=3,
            persistence_threshold=0.5,
            drift_threshold=0.3,
        )
        winners = list(range(100))
        # Build up to stable
        for i in range(5):
            tracker.update("A", winners, step=i)
        assert tracker.assemblies["A:0"].status == "stable"

        # Drift: 70/130 ≈ 0.538 overlap — above persistence but below convergence
        drifted = list(range(70)) + list(range(200, 230))
        tracker.update("A", drifted, step=5)
        asm = tracker.assemblies["A:0"]
        assert asm.status == "drifting"
        assert asm.consecutive_stable == 0

    def test_recovery_from_drift(self):
        tracker = AssemblyTracker(
            convergence_threshold=0.8,
            convergence_steps=2,
            persistence_threshold=0.5,
            drift_threshold=0.3,
        )
        winners = list(range(100))
        for i in range(3):
            tracker.update("A", winners, step=i)
        # Drift: 70/130 overlap
        drifted = list(range(70)) + list(range(200, 230))
        tracker.update("A", drifted, step=3)
        assert tracker.assemblies["A:0"].status == "drifting"
        # Recover: back to original winners
        for i in range(4, 7):
            tracker.update("A", winners, step=i)
        asm = tracker.assemblies["A:0"]
        # After drift, canonical was NOT updated (overlap < convergence)
        # but the new winners match closely, so it should re-stabilize
        assert asm.status in ("stable", "forming")


class TestLoss:
    def test_lost_on_no_winners(self):
        tracker = AssemblyTracker()
        tracker.update("A", list(range(100)), step=0)
        tracker.update("A", [], step=1)
        asm = tracker.assemblies["A:0"]
        assert asm.status == "lost"

    def test_lost_on_completely_new_winners(self):
        tracker = AssemblyTracker(
            persistence_threshold=0.6, drift_threshold=0.4
        )
        tracker.update("A", list(range(100)), step=0)
        # Completely different winners → below drift threshold
        tracker.update("A", list(range(500, 600)), step=1)
        # Should create a new assembly since overlap < drift_threshold
        active = tracker.get_active_assemblies("A")
        # Original assembly should be lost, new one should exist
        assert len(tracker.assemblies) >= 2


class TestProcessLog:
    def test_process_simple_log(self):
        steps = [
            {"step": 0, "activations": {
                "A": {"winners": list(range(100))},
                "B": {"winners": list(range(50))},
            }},
            {"step": 1, "activations": {
                "A": {"winners": list(range(100))},
                "B": {"winners": list(range(50))},
            }},
        ]
        tracker = AssemblyTracker()
        mapping = tracker.process_log(steps)
        assert 0 in mapping
        assert 1 in mapping
        assert "A" in mapping[0]
        assert "B" in mapping[0]

    def test_process_log_tracks_ids(self):
        winners = list(range(100))
        steps = [
            {"step": i, "activations": {"A": {"winners": winners}}}
            for i in range(5)
        ]
        tracker = AssemblyTracker()
        mapping = tracker.process_log(steps)
        # Same assembly across all steps
        ids = [mapping[i]["A"] for i in range(5)]
        assert len(set(ids)) == 1


class TestQueries:
    def test_get_assembly(self):
        tracker = AssemblyTracker()
        tracker.update("A", list(range(100)), step=0)
        asm = tracker.get_assembly("A:0")
        assert asm is not None
        assert asm.area == "A"

    def test_get_nonexistent(self):
        tracker = AssemblyTracker()
        assert tracker.get_assembly("nope") is None

    def test_get_active_assemblies(self):
        tracker = AssemblyTracker()
        tracker.update("A", list(range(100)), step=0)
        tracker.update("B", list(range(50)), step=0)
        active = tracker.get_active_assemblies()
        assert len(active) == 2

    def test_get_active_by_area(self):
        tracker = AssemblyTracker()
        tracker.update("A", list(range(100)), step=0)
        tracker.update("B", list(range(50)), step=0)
        active = tracker.get_active_assemblies("A")
        assert len(active) == 1
        assert active[0].area == "A"

    def test_get_area_history(self):
        tracker = AssemblyTracker()
        tracker.update("A", list(range(100)), step=0)
        tracker.update("A", [], step=1)  # loses A
        tracker.update("A", list(range(200, 300)), step=2)  # new assembly
        history = tracker.get_area_history("A")
        assert len(history) >= 2

    def test_summary(self):
        tracker = AssemblyTracker()
        tracker.update("A", list(range(100)), step=0)
        summary = tracker.get_summary()
        assert len(summary) == 1
        assert summary[0]["id"] == "A:0"
        assert summary[0]["canonical_size"] == 100


class TestJaccard:
    def test_identical(self):
        tracker = AssemblyTracker()
        assert tracker._jaccard({1, 2, 3}, {1, 2, 3}) == 1.0

    def test_disjoint(self):
        tracker = AssemblyTracker()
        assert tracker._jaccard({1, 2}, {3, 4}) == 0.0

    def test_partial(self):
        tracker = AssemblyTracker()
        # {1,2,3} ∩ {2,3,4} = {2,3}, union = {1,2,3,4}
        assert tracker._jaccard({1, 2, 3}, {2, 3, 4}) == pytest.approx(0.5)

    def test_empty_both(self):
        tracker = AssemblyTracker()
        assert tracker._jaccard(set(), set()) == 1.0

    def test_empty_one(self):
        tracker = AssemblyTracker()
        assert tracker._jaccard(set(), {1, 2}) == 0.0
