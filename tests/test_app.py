"""Tests for viz/app.py — Live Interactive Sandbox."""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from viz.brain_api import LiveBrain
from viz.app import create_app, _render_graph, _render_area_table, _render_log
from viz.app import _render_convergence, _render_weight_histogram
from viz.app import _render_neuron_detail


# ── App Creation ────────────────────────────────────────────────────────────

class TestAppCreation:
    def test_create_app(self):
        brain = LiveBrain(p=0.1, seed=1)
        app = create_app(brain)
        assert app is not None

    def test_app_has_layout(self):
        brain = LiveBrain(p=0.1, seed=1)
        app = create_app(brain)
        assert app.layout is not None

    def test_create_with_prebuilt_brain(self):
        brain = LiveBrain(p=0.1, seed=1)
        brain.add_area("X", n=100, k=10)
        app = create_app(brain)
        assert app is not None


# ── Render Helpers ──────────────────────────────────────────────────────────

class TestRenderGraph:
    def test_empty_brain(self):
        brain = LiveBrain(p=0.1, seed=1)
        create_app(brain)
        result = _render_graph([])
        # Should return placeholder
        assert hasattr(result, 'children') or result is not None

    def test_with_areas(self):
        brain = LiveBrain(p=0.1, seed=1)
        brain.add_area("A", n=100, k=10)
        brain.add_area("B", n=100, k=10)
        create_app(brain)
        result = _render_graph(["A", "B"])
        # Should be a Cytoscape component
        assert result is not None

    def test_active_area_styling(self):
        brain = LiveBrain(p=0.1, seed=1)
        brain.add_area("A", n=100, k=10)
        brain.add_stimulus("s", size=10)
        brain.stimulate("s", "A")
        create_app(brain)
        result = _render_graph(["A"])
        assert result is not None


class TestRenderAreaTable:
    def test_empty(self):
        brain = LiveBrain(p=0.1, seed=1)
        create_app(brain)
        result = _render_area_table()
        # Should return "No areas yet" div
        assert result is not None

    def test_with_data(self):
        brain = LiveBrain(p=0.1, seed=1)
        brain.add_area("A", n=100, k=10)
        brain.add_stimulus("s", size=10)
        brain.stimulate("s", "A")
        create_app(brain)
        result = _render_area_table()
        assert result is not None


class TestRenderLog:
    def test_empty(self):
        brain = LiveBrain(p=0.1, seed=1)
        create_app(brain)
        result = _render_log()
        assert result is not None

    def test_with_operations(self):
        brain = LiveBrain(p=0.1, seed=1)
        brain.add_area("A", n=100, k=10)
        brain.add_area("B", n=100, k=10)
        brain.add_stimulus("s", size=10)
        brain.stimulate("s", "A")
        brain.project("A", "B")
        create_app(brain)
        result = _render_log()
        assert result is not None


class TestConvergenceChart:
    def test_no_area(self):
        brain = LiveBrain(p=0.1, seed=1)
        create_app(brain)
        fig = _render_convergence(None)
        assert fig is not None

    def test_with_area(self):
        brain = LiveBrain(p=0.1, seed=1)
        brain.add_area("A", n=100, k=10)
        brain.add_stimulus("s", size=10)
        for _ in range(5):
            brain.stimulate("s", "A")
        create_app(brain)
        fig = _render_convergence("A")
        assert len(fig.data) >= 1  # at least overlap trace


class TestWeightHistogram:
    def test_no_areas(self):
        brain = LiveBrain(p=0.1, seed=1)
        create_app(brain)
        fig = _render_weight_histogram(None, None)
        assert fig is not None

    def test_with_areas(self):
        brain = LiveBrain(p=0.1, seed=1)
        brain.add_area("A", n=100, k=10)
        brain.add_area("B", n=100, k=10)
        create_app(brain)
        fig = _render_weight_histogram("A", "B")
        assert fig is not None


# ── Integration Tests ───────────────────────────────────────────────────────

class TestIntegration:
    def test_full_workflow(self):
        """Simulate a typical user workflow."""
        brain = LiveBrain(p=0.1, seed=42)
        create_app(brain)

        # Add areas
        brain.add_area("A", n=500, k=50)
        brain.add_area("B", n=500, k=50)

        # Add stimulus
        brain.add_stimulus("input", size=50)

        # Stimulate A to form assembly
        for _ in range(10):
            brain.stimulate("input", "A")
        state_a = brain.get_area_state("A")
        assert state_a["n_winners"] == 50
        assert state_a["overlap_prev"] > 0.5  # some convergence

        # Project A→B to copy assembly
        for _ in range(10):
            brain.project("A", "B")
        state_b = brain.get_area_state("B")
        assert state_b["n_winners"] > 0

        # Render should work
        graph = _render_graph(["A", "B"])
        table = _render_area_table()
        conv = _render_convergence("A")
        wt = _render_weight_histogram("A", "B")
        assert all(x is not None for x in [graph, table, conv, wt])

    def test_learning_toggle_workflow(self):
        """Test learning ON/OFF doesn't crash."""
        brain = LiveBrain(p=0.1, seed=42)
        create_app(brain)

        brain.add_area("A", n=200, k=20)
        brain.add_stimulus("s", size=20)
        brain.stimulate("s", "A", rounds=5)

        # Toggle off
        brain.set_learning(False)
        brain.stimulate("s", "A", rounds=3)

        # Toggle back on
        brain.set_learning(True)
        brain.stimulate("s", "A", rounds=3)

        assert len(brain.steps) >= 5

    def test_convergence_history_matches_steps(self):
        """Overlap history length should match number of log steps."""
        brain = LiveBrain(p=0.1, seed=42)
        brain.add_area("A", n=200, k=20)
        brain.add_stimulus("s", size=20)
        for _ in range(5):
            brain.stimulate("s", "A")
        create_app(brain)

        history = brain.get_overlap_history("A")
        assert len(history) == len(brain.steps)


class TestNeuronDetail:
    def test_no_area_selected(self):
        brain = LiveBrain(p=0.1, seed=1)
        create_app(brain)
        result = _render_neuron_detail(None)
        assert result is not None

    def test_area_no_assembly(self):
        brain = LiveBrain(p=0.1, seed=1)
        brain.add_area("A", n=100, k=10)
        create_app(brain)
        result = _render_neuron_detail("A")
        assert result is not None

    def test_area_with_winners(self):
        brain = LiveBrain(p=0.1, seed=1)
        brain.add_area("A", n=100, k=10)
        brain.add_stimulus("s", size=10)
        brain.stimulate("s", "A")
        create_app(brain)
        result = _render_neuron_detail("A")
        assert result is not None

    def test_nonexistent_area(self):
        brain = LiveBrain(p=0.1, seed=1)
        create_app(brain)
        result = _render_neuron_detail("NOPE")
        assert result is not None


class TestGainedLostDisplay:
    def test_first_stimulate_shows_gained_in_table(self):
        """Area table should show non-zero gained after first stimulate."""
        brain = LiveBrain(p=0.1, seed=42)
        brain.add_area("A", n=200, k=20)
        brain.add_stimulus("s", size=20)
        brain.stimulate("s", "A")
        create_app(brain)
        # The _last_snapshot should have correct data
        snap = brain._last_snapshot["A"]
        assert snap["gained"] == 20
        assert snap["lost"] == 0

    def test_table_renders_with_snapshot_data(self):
        brain = LiveBrain(p=0.1, seed=42)
        brain.add_area("A", n=200, k=20)
        brain.add_stimulus("s", size=20)
        brain.stimulate("s", "A")
        create_app(brain)
        table = _render_area_table()
        assert table is not None
