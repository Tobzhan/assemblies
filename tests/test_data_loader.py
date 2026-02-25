"""Tests for viz/data_loader.py."""

import json
import os
import tempfile
import pytest

# Add parent dir to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from viz.data_loader import SimulationData, load_simulation


def _make_log_file(steps: list[dict]) -> str:
    """Write steps to a temp JSONL file and return the path."""
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "simulation_log.jsonl")
    with open(path, "w") as f:
        for step in steps:
            f.write(json.dumps(step) + "\n")
    return path


SAMPLE_STEPS = [
    {
        "step": 0,
        "phase": "init",
        "dfa_state": "q0",
        "dfa_symbol": None,
        "dfa_transition": None,
        "activations": {
            "q0": {
                "area": "q0",
                "winners": list(range(100)),
                "winner_inputs": [1.0] * 100,
                "num_new_winners": 0,
                "activation_margin": 0,
                "attribution": {},
                "template_name": "q0",
                "template_overlap": 1.0,
                "winners_gained": list(range(100)),
                "winners_lost": [],
                "overlap_with_prev": 0.0,
                "input_stats": None,
            },
            "q1": {
                "area": "q1",
                "winners": [],
                "winner_inputs": [],
                "num_new_winners": 0,
                "activation_margin": 0,
                "attribution": {},
                "template_name": None,
                "template_overlap": 0.0,
                "winners_gained": [],
                "winners_lost": [],
                "overlap_with_prev": 0.0,
                "input_stats": None,
            },
        },
    },
    {
        "step": 1,
        "phase": "project",
        "dfa_state": "q1",
        "dfa_symbol": "a",
        "dfa_transition": "(q0, a) → q1",
        "activations": {
            "q0": {
                "area": "q0",
                "winners": [],
                "winner_inputs": [],
                "num_new_winners": 0,
                "activation_margin": 0,
                "attribution": {"Transitions": 0.0},
                "template_name": "q0",
                "template_overlap": 0.0,
                "winners_gained": [],
                "winners_lost": list(range(100)),
                "overlap_with_prev": 0.0,
                "input_stats": None,
            },
            "q1": {
                "area": "q1",
                "winners": list(range(100)),
                "winner_inputs": [5.0] * 100,
                "num_new_winners": 0,
                "activation_margin": 5.0,
                "attribution": {"Transitions": 500.0},
                "template_name": "q1",
                "template_overlap": 1.0,
                "winners_gained": list(range(100)),
                "winners_lost": [],
                "overlap_with_prev": 0.0,
                "input_stats": {"min": 0+0, "max": 5.0, "mean": 0.5, "std": 1.5,
                                "median": 0.0, "p25": 0.0, "p75": 0.0},
            },
            "Transitions": {
                "area": "Transitions",
                "winners": list(range(100)),
                "winner_inputs": [10.0] * 100,
                "num_new_winners": 0,
                "activation_margin": 10.0,
                "attribution": {"state_q0": 500.0, "stim_a": 500.0},
                "template_name": "trans_q0_a_q1",
                "template_overlap": 1.0,
                "winners_gained": list(range(100)),
                "winners_lost": [],
                "overlap_with_prev": 0.0,
                "input_stats": {"min": 0+0, "max": 10.0, "mean": 1.67, "std": 3.5,
                                "median": 0.0, "p25": 0.0, "p75": 0.0},
            },
        },
    },
]


class TestLoadSimulation:
    def test_load_basic(self):
        path = _make_log_file(SAMPLE_STEPS)
        data = load_simulation(path)
        assert data.num_steps == 2

    def test_load_empty_file(self):
        path = _make_log_file([])
        data = load_simulation(path)
        assert data.num_steps == 0


class TestSimulationData:
    def test_get_step(self):
        data = SimulationData(SAMPLE_STEPS)
        step = data.get_step(0)
        assert step["dfa_state"] == "q0"

    def test_get_step_out_of_bounds(self):
        data = SimulationData(SAMPLE_STEPS)
        assert data.get_step(999) == {}
        assert data.get_step(-1) == {}

    def test_get_area_names(self):
        data = SimulationData(SAMPLE_STEPS)
        areas = data.get_area_names()
        assert "q0" in areas
        assert "q1" in areas
        assert "Transitions" in areas

    def test_get_dfa_path(self):
        data = SimulationData(SAMPLE_STEPS)
        path = data.get_dfa_path()
        assert len(path) == 2
        assert path[0]["state"] == "q0"
        assert path[1]["state"] == "q1"
        assert path[1]["symbol"] == "a"
        assert path[1]["transition"] == "(q0, a) → q1"

    def test_get_area_activation(self):
        data = SimulationData(SAMPLE_STEPS)
        act = data.get_area_activation(1, "q1")
        assert act["template_overlap"] == 1.0
        assert len(act["winners"]) == 100

    def test_get_overlap_series(self):
        data = SimulationData(SAMPLE_STEPS)
        series = data.get_overlap_series("q1")
        assert series == [0.0, 1.0]

    def test_get_margin_series(self):
        data = SimulationData(SAMPLE_STEPS)
        series = data.get_margin_series("Transitions")
        assert series == [0.0, 10.0]

    def test_get_winner_count_series(self):
        data = SimulationData(SAMPLE_STEPS)
        series = data.get_winner_count_series("q0")
        assert series == [100, 0]


class TestBrainThought:
    def test_init_thought(self):
        data = SimulationData(SAMPLE_STEPS)
        thought = data.generate_thought(0)
        assert "Initialization" in thought
        assert "q0" in thought

    def test_project_thought(self):
        data = SimulationData(SAMPLE_STEPS)
        thought = data.generate_thought(1)
        assert "Projection" in thought
        assert "Symbol 'a'" in thought
        assert "(q0, a)" in thought
        assert "q1" in thought

    def test_attribution_in_thought(self):
        data = SimulationData(SAMPLE_STEPS)
        thought = data.generate_thought(1)
        assert "state_q0" in thought or "stim_a" in thought

    def test_empty_step(self):
        data = SimulationData(SAMPLE_STEPS)
        thought = data.generate_thought(999)
        assert "No data" in thought


class TestDiff:
    def test_diff_with_changes(self):
        data = SimulationData(SAMPLE_STEPS)
        diff = data.generate_diff(1)
        assert "Diff" in diff
        assert "q0" in diff
        assert "gained" in diff or "lost" in diff or "no change" in diff

    def test_diff_empty_step(self):
        data = SimulationData(SAMPLE_STEPS)
        diff = data.generate_diff(999)
        assert "No data" in diff
