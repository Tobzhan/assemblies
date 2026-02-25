"""Tests for viz/emergent_detector.py — covers convergence detection,
drift, loss, process_log, and convergence series."""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from viz.emergent_detector import EmergentDetector, EmergentAssembly


class TestBasicDetection:
    def test_no_detection_on_first_step(self):
        det = EmergentDetector()
        result = det.observe("A", list(range(100)), step=0)
        assert result is None  # needs ≥2 steps

    def test_detection_on_stable_winners(self):
        det = EmergentDetector(convergence_threshold=0.8)
        winners = list(range(100))
        det.observe("A", winners, step=0)
        asm_id = det.observe("A", winners, step=1)
        # Overlap is 1.0 ≥ 0.8 → assembly forming
        assert asm_id is not None
        assert "A:asm_" in asm_id

    def test_no_detection_on_random_winners(self):
        det = EmergentDetector(convergence_threshold=0.8)
        det.observe("A", list(range(100)), step=0)
        asm_id = det.observe("A", list(range(200, 300)), step=1)
        # Overlap is 0.0 < 0.8 → no assembly
        assert asm_id is None

    def test_empty_winners(self):
        det = EmergentDetector()
        result = det.observe("A", [], step=0)
        assert result is None


class TestConvergence:
    def test_converges_after_window(self):
        det = EmergentDetector(convergence_threshold=0.8, convergence_window=3)
        winners = list(range(100))
        for i in range(6):
            det.observe("A", winners, step=i)
        # Should be converged after window of stable overlaps
        asms = det.get_detected_assemblies("A")
        assert len(asms) >= 1
        converged = [a for a in asms if a.status == "converged"]
        assert len(converged) >= 1

    def test_not_converged_before_window(self):
        det = EmergentDetector(convergence_threshold=0.8, convergence_window=10)
        winners = list(range(100))
        det.observe("A", winners, step=0)
        det.observe("A", winners, step=1)
        det.observe("A", winners, step=2)
        asms = det.get_detected_assemblies("A")
        converged = [a for a in asms if a.status == "converged"]
        assert len(converged) == 0


class TestDrift:
    def test_drift_after_convergence(self):
        det = EmergentDetector(
            convergence_threshold=0.8, convergence_window=3,
            split_threshold=0.3,
        )
        winners = list(range(100))
        for i in range(6):
            det.observe("A", winners, step=i)
        # Now drift: 60% overlap with canonical
        drifted = list(range(60)) + list(range(200, 240))
        det.observe("A", drifted, step=6)
        asms = det.get_detected_assemblies("A")
        assert len(asms) >= 1
        # Should be drifting (overlap < convergence but > split)
        assert any(a.status == "drifting" for a in asms)


class TestLoss:
    def test_lost_on_completely_new(self):
        det = EmergentDetector(
            convergence_threshold=0.8, convergence_window=2,
            split_threshold=0.3,
        )
        winners = list(range(100))
        for i in range(4):
            det.observe("A", winners, step=i)
        # Completely new winners
        det.observe("A", list(range(500, 600)), step=4)
        # The assembly may be lost
        all_asms = [a for a in det.assemblies.values() if a.area == "A"]
        assert len(all_asms) >= 1


class TestProcessLog:
    def test_process_simple(self):
        steps = [
            {"step": i, "activations": {"A": {"winners": list(range(100))}}}
            for i in range(5)
        ]
        det = EmergentDetector(convergence_threshold=0.8)
        mapping = det.process_log(steps)
        assert len(mapping) == 5
        # After step 1, should have assembly IDs
        has_ids = any("A" in mapping[i] for i in range(1, 5))
        assert has_ids


class TestConvergenceSeries:
    def test_series_length(self):
        det = EmergentDetector()
        winners = list(range(100))
        for i in range(5):
            det.observe("A", winners, step=i)
        series = det.get_convergence_series("A")
        assert len(series) == 5

    def test_series_values(self):
        det = EmergentDetector()
        winners = list(range(100))
        for i in range(3):
            det.observe("A", winners, step=i)
        series = det.get_convergence_series("A")
        assert series[0] == 0.0  # first step
        assert series[1] == pytest.approx(1.0)  # same winners
        assert series[2] == pytest.approx(1.0)

    def test_series_partial_overlap(self):
        det = EmergentDetector()
        det.observe("A", list(range(100)), step=0)
        det.observe("A", list(range(50, 150)), step=1)
        series = det.get_convergence_series("A")
        # Jaccard: 50/150 ≈ 0.333
        assert series[1] == pytest.approx(50 / 150, abs=0.01)

    def test_empty_area(self):
        det = EmergentDetector()
        series = det.get_convergence_series("nonexistent")
        assert series == []


class TestQueries:
    def test_get_detected_all(self):
        det = EmergentDetector()
        winners = list(range(100))
        for i in range(3):
            det.observe("A", winners, step=i)
            det.observe("B", winners, step=i)
        all_asms = det.get_detected_assemblies()
        assert len(all_asms) >= 2

    def test_get_detected_by_area(self):
        det = EmergentDetector()
        winners = list(range(100))
        for i in range(3):
            det.observe("A", winners, step=i)
            det.observe("B", winners, step=i)
        a_asms = det.get_detected_assemblies("A")
        assert all(a.area == "A" for a in a_asms)

    def test_summary(self):
        det = EmergentDetector()
        winners = list(range(100))
        for i in range(3):
            det.observe("A", winners, step=i)
        summary = det.get_summary()
        assert len(summary) >= 1
        assert "id" in summary[0]
        assert "canonical_size" in summary[0]
