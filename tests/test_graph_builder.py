"""Tests for viz/graph_builder.py — covers all 3 graph levels
and component creation."""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from viz.graph_builder import (
    build_area_graph, build_assembly_view, build_neuron_view,
    create_graph_component, create_graph_panel,
    _calculate_positions, _infer_edges,
)


# ── Test data ────────────────────────────────────────────────────────────────

STEP_INIT = {
    "step": 0, "phase": "init", "dfa_state": "q0",
    "dfa_symbol": None, "dfa_transition": None,
    "activations": {
        "q0": {
            "winners": list(range(100)), "template_overlap": 1.0,
            "template_name": "q0", "num_new_winners": 100,
            "winners_gained": list(range(100)), "winners_lost": [],
            "activation_margin": 10.0, "attribution": {"stim_init": 500.0},
        },
    },
}

STEP_TRANSITION = {
    "step": 1, "phase": "project", "dfa_state": "q1",
    "dfa_symbol": "a", "dfa_transition": "(q0, a) → q1",
    "activations": {
        "q0": {
            "winners": [], "template_overlap": 0.0,
            "template_name": "q0", "num_new_winners": 0,
            "winners_gained": [], "winners_lost": list(range(100)),
            "activation_margin": 0, "attribution": {},
        },
        "q1": {
            "winners": list(range(100)), "template_overlap": 1.0,
            "template_name": "q1", "num_new_winners": 0,
            "winners_gained": list(range(100)), "winners_lost": [],
            "activation_margin": 5.0,
            "attribution": {"from_Transitions": 500.0},
        },
        "Transitions": {
            "winners": list(range(100)), "template_overlap": 1.0,
            "template_name": "trans_q0_a_q1", "num_new_winners": 0,
            "winners_gained": list(range(100)), "winners_lost": [],
            "activation_margin": 10.0,
            "attribution": {"from_q0": 500.0, "stim_a": 500.0},
        },
    },
}

STEP_DRIFT = {
    "step": 2, "phase": "project", "dfa_state": "q1",
    "dfa_symbol": "b", "dfa_transition": "(q1, b) → q1",
    "activations": {
        "q1": {
            "winners": list(range(80)) + list(range(200, 220)),
            "template_overlap": 0.72, "template_name": "q1",
            "num_new_winners": 20,
            "winners_gained": list(range(200, 220)),
            "winners_lost": list(range(80, 100)),
            "activation_margin": 1.5,
            "attribution": {"from_Transitions": 400.0},
        },
    },
}


# ── Level 0: Area Graph ─────────────────────────────────────────────────────

class TestAreaGraph:
    def test_init_step(self):
        elements = build_area_graph(STEP_INIT)
        nodes = [e for e in elements if "source" not in e.get("data", {})]
        assert len(nodes) >= 1
        # q0 should be current state
        q0 = next(e for e in nodes if e["data"]["id"] == "q0")
        assert "current-state" in q0["classes"]

    def test_transition_step(self):
        elements = build_area_graph(STEP_TRANSITION)
        nodes = [e for e in elements if "source" not in e.get("data", {})]
        edges = [e for e in elements if "source" in e.get("data", {})]
        assert len(nodes) >= 3  # q0, q1, Transitions
        assert len(edges) >= 1  # at least one edge

    def test_dormant_area(self):
        elements = build_area_graph(STEP_TRANSITION)
        q0 = next(e for e in elements
                  if e.get("data", {}).get("id") == "q0")
        assert "dormant" in q0["classes"]

    def test_custom_area_names(self):
        elements = build_area_graph(STEP_INIT, area_names=["q0", "q1"])
        nodes = [e for e in elements if "source" not in e.get("data", {})]
        assert len(nodes) == 2

    def test_empty_step(self):
        elements = build_area_graph({"activations": {}})
        assert elements == []


# ── Level 1: Assembly View ───────────────────────────────────────────────────

class TestAssemblyView:
    def test_basic(self):
        elements = build_assembly_view("q1", STEP_TRANSITION)
        assert len(elements) >= 1
        # Should have main assembly node
        main = next(e for e in elements
                    if e["data"]["id"] == "q1_main")
        assert "q1" in main["data"]["label"]

    def test_with_gained_lost(self):
        elements = build_assembly_view("q1", STEP_DRIFT)
        # Should have gained and lost sub-nodes
        ids = [e["data"]["id"] for e in elements
               if "source" not in e.get("data", {})]
        assert "q1_gained" in ids
        assert "q1_lost" in ids

    def test_with_attribution(self):
        elements = build_assembly_view("Transitions", STEP_TRANSITION)
        # Should have attribution source nodes
        src_nodes = [e for e in elements
                     if e.get("data", {}).get("id", "").startswith("Transitions_src_")]
        assert len(src_nodes) >= 2  # from_q0 + stim_a

    def test_missing_area(self):
        elements = build_assembly_view("nonexistent", STEP_INIT)
        assert elements == []


# ── Level 2: Neuron View ─────────────────────────────────────────────────────

class TestNeuronView:
    def test_basic(self):
        elements = build_neuron_view("q1", STEP_TRANSITION)
        assert len(elements) >= 100  # 100 winners

    def test_with_gained_lost(self):
        elements = build_neuron_view("q1", STEP_DRIFT)
        classes_str = " ".join(
            e.get("classes", "") for e in elements
        )
        assert "neuron-gained" in classes_str
        assert "neuron-lost" in classes_str

    def test_max_neurons_limit(self):
        elements = build_neuron_view("q1", STEP_TRANSITION, max_neurons=50)
        nodes = [e for e in elements if "source" not in e.get("data", {})]
        assert len(nodes) <= 50

    def test_missing_area(self):
        elements = build_neuron_view("nonexistent", STEP_INIT)
        assert elements == []


# ── Component creation ───────────────────────────────────────────────────────

class TestComponentCreation:
    def test_create_graph_component(self):
        elements = build_area_graph(STEP_INIT)
        component = create_graph_component(elements)
        assert component is not None
        assert component.id == "cyto-graph"

    def test_create_graph_panel(self):
        panel = create_graph_panel(STEP_TRANSITION, step_idx=1, level=0)
        assert panel is not None

    def test_panel_level_1(self):
        panel = create_graph_panel(
            STEP_DRIFT, step_idx=2, selected_area="q1", level=1
        )
        assert panel is not None

    def test_panel_level_2(self):
        panel = create_graph_panel(
            STEP_TRANSITION, step_idx=1, selected_area="q1", level=2
        )
        assert panel is not None


# ── Helpers ──────────────────────────────────────────────────────────────────

class TestHelpers:
    def test_calculate_positions(self):
        positions = _calculate_positions(["q0", "q1", "Transitions"])
        assert "q0" in positions
        assert "q1" in positions
        assert "Transitions" in positions

    def test_infer_edges(self):
        edges = _infer_edges(STEP_TRANSITION, ["q0", "q1", "Transitions"])
        assert len(edges) >= 1
        # At least one active edge during transition
        active = [e for e in edges if e[3]]
        assert len(active) >= 1
