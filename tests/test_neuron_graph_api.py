"""Tests for brain_api neuron graph data methods."""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from viz.brain_api import LiveBrain


# ── get_neuron_graph_data ────────────────────────────────────────────────────

class TestNeuronGraphData:
    def test_basic_structure(self):
        lb = LiveBrain(p=0.1, seed=42)
        lb.add_area("A", n=100, k=10)
        data = lb.get_neuron_graph_data("A")
        assert "nodes" in data
        assert "edges" in data
        assert "meta" in data

    def test_node_count_matches_area(self):
        lb = LiveBrain(p=0.1, seed=42)
        lb.add_area("A", n=100, k=10)
        data = lb.get_neuron_graph_data("A")
        assert len(data["nodes"]) == 100

    def test_node_states_before_stimulation(self):
        lb = LiveBrain(p=0.1, seed=42)
        lb.add_area("A", n=50, k=5)
        data = lb.get_neuron_graph_data("A")
        states = [n["state"] for n in data["nodes"]]
        # No winners yet, all should be inactive
        assert all(s == "inactive" for s in states)

    def test_node_states_after_stimulation(self):
        lb = LiveBrain(p=0.1, seed=42)
        lb.add_area("A", n=100, k=10)
        lb.add_stimulus("s", size=10)
        lb.stimulate("s", "A")
        data = lb.get_neuron_graph_data("A")
        states = [n["state"] for n in data["nodes"]]
        winner_count = sum(1 for s in states if s == "winner" or s == "gained")
        assert winner_count == 10

    def test_gained_state_on_first_stimulate(self):
        lb = LiveBrain(p=0.1, seed=42)
        lb.add_area("A", n=100, k=10)
        lb.add_stimulus("s", size=10)
        lb.stimulate("s", "A")
        data = lb.get_neuron_graph_data("A")
        gained = [n for n in data["nodes"] if n["state"] == "gained"]
        assert len(gained) == 10  # all new winners are "gained"

    def test_edges_present_after_stimulation(self):
        lb = LiveBrain(p=0.1, seed=42)
        lb.add_area("A", n=100, k=10)
        lb.add_stimulus("s", size=10)
        lb.stimulate("s", "A")
        data = lb.get_neuron_graph_data("A")
        assert len(data["edges"]) > 0

    def test_edge_weight_is_positive(self):
        lb = LiveBrain(p=0.1, seed=42)
        lb.add_area("A", n=100, k=10)
        lb.add_stimulus("s", size=10)
        lb.stimulate("s", "A")
        data = lb.get_neuron_graph_data("A")
        for edge in data["edges"]:
            assert edge["weight"] > 0

    def test_meta_contains_required_fields(self):
        lb = LiveBrain(p=0.1, seed=42)
        lb.add_area("A", n=100, k=10)
        data = lb.get_neuron_graph_data("A")
        meta = data["meta"]
        assert meta["area"] == "A"
        assert meta["n"] == 100
        assert meta["k"] == 10

    def test_nonexistent_area(self):
        lb = LiveBrain(p=0.1, seed=42)
        data = lb.get_neuron_graph_data("NOPE")
        assert data["nodes"] == []
        assert data["edges"] == []
        assert data["meta"] == {}

    def test_winners_only_filter(self):
        lb = LiveBrain(p=0.1, seed=42)
        lb.add_area("A", n=100, k=10)
        lb.add_stimulus("s", size=10)
        lb.stimulate("s", "A")

        # With filter (default)
        data_filtered = lb.get_neuron_graph_data("A", winners_only_edges=True)
        # Without filter
        data_all = lb.get_neuron_graph_data("A", winners_only_edges=False)

        # Unfiltered should have >= filtered edges
        assert len(data_all["edges"]) >= len(data_filtered["edges"])

    def test_self_edges_excluded(self):
        lb = LiveBrain(p=0.1, seed=42)
        lb.add_area("A", n=100, k=10)
        lb.add_stimulus("s", size=10)
        lb.stimulate("s", "A")
        data = lb.get_neuron_graph_data("A")
        for edge in data["edges"]:
            assert edge["source"] != edge["target"]


# ── get_cross_area_graph_data ────────────────────────────────────────────────

class TestCrossAreaGraphData:
    def test_basic_structure(self):
        lb = LiveBrain(p=0.1, seed=42)
        lb.add_area("A", n=100, k=10)
        lb.add_area("B", n=100, k=10)
        data = lb.get_cross_area_graph_data("A", "B")
        assert "nodes_a" in data
        assert "nodes_b" in data
        assert "edges_ab" in data
        assert "edges_ba" in data
        assert "meta" in data

    def test_node_counts(self):
        lb = LiveBrain(p=0.1, seed=42)
        lb.add_area("A", n=100, k=10)
        lb.add_area("B", n=200, k=20)
        data = lb.get_cross_area_graph_data("A", "B")
        assert len(data["nodes_a"]) == 100
        assert len(data["nodes_b"]) == 200

    def test_edges_after_projection(self):
        lb = LiveBrain(p=0.1, seed=42)
        lb.add_area("A", n=100, k=10)
        lb.add_area("B", n=100, k=10)
        lb.add_stimulus("s", size=10)
        lb.stimulate("s", "A", rounds=5)
        lb.project("A", "B", rounds=5)
        data = lb.get_cross_area_graph_data("A", "B")
        # A→B edges should exist after projection
        assert len(data["edges_ab"]) > 0

    def test_meta_fields(self):
        lb = LiveBrain(p=0.1, seed=42)
        lb.add_area("A", n=100, k=10)
        lb.add_area("B", n=100, k=10)
        data = lb.get_cross_area_graph_data("A", "B")
        meta = data["meta"]
        assert meta["area_a"] == "A"
        assert meta["area_b"] == "B"
        assert meta["n_a"] == 100
        assert meta["n_b"] == 100

    def test_nonexistent_area(self):
        lb = LiveBrain(p=0.1, seed=42)
        lb.add_area("A", n=100, k=10)
        data = lb.get_cross_area_graph_data("A", "NOPE")
        assert data["nodes_a"] == []
        assert data["edges_ab"] == []

    def test_bidirectional_edges(self):
        """After reciprocal projection, both directions should have edges."""
        lb = LiveBrain(p=0.1, seed=42)
        lb.add_area("A", n=100, k=10)
        lb.add_area("B", n=100, k=10)
        lb.add_stimulus("s", size=10)
        lb.stimulate("s", "A", rounds=5)
        lb.project("A", "B", rounds=5)
        lb.reciprocal_project("A", "B", rounds=5)
        data = lb.get_cross_area_graph_data("A", "B")
        assert len(data["edges_ab"]) > 0
        assert len(data["edges_ba"]) > 0

    def test_edge_weight_positive(self):
        lb = LiveBrain(p=0.1, seed=42)
        lb.add_area("A", n=100, k=10)
        lb.add_area("B", n=100, k=10)
        lb.add_stimulus("s", size=10)
        lb.stimulate("s", "A", rounds=5)
        lb.project("A", "B", rounds=5)
        data = lb.get_cross_area_graph_data("A", "B")
        for edge in data["edges_ab"]:
            assert edge["weight"] > 0


# ── get_area_edge_summary ────────────────────────────────────────────────────

class TestAreaEdgeSummary:
    def test_empty_brain(self):
        lb = LiveBrain(p=0.1, seed=42)
        summary = lb.get_area_edge_summary()
        assert summary == {}

    def test_single_area(self):
        lb = LiveBrain(p=0.1, seed=42)
        lb.add_area("A", n=100, k=10)
        summary = lb.get_area_edge_summary()
        # Single area has no pairs
        assert summary == {}

    def test_two_areas(self):
        lb = LiveBrain(p=0.1, seed=42)
        lb.add_area("A", n=100, k=10)
        lb.add_area("B", n=100, k=10)
        summary = lb.get_area_edge_summary()
        assert len(summary) == 1  # one pair
        key = "A__B"
        assert key in summary

    def test_three_areas(self):
        lb = LiveBrain(p=0.1, seed=42)
        lb.add_area("A", n=100, k=10)
        lb.add_area("B", n=100, k=10)
        lb.add_area("C", n=100, k=10)
        summary = lb.get_area_edge_summary()
        assert len(summary) == 3  # A-B, A-C, B-C

    def test_summary_fields(self):
        lb = LiveBrain(p=0.1, seed=42)
        lb.add_area("A", n=100, k=10)
        lb.add_area("B", n=100, k=10)
        summary = lb.get_area_edge_summary()
        entry = summary["A__B"]
        assert "mean_weight" in entry
        assert "max_weight" in entry
        assert "pct_strengthened" in entry
        assert "total_connections" in entry
        assert "area_a" in entry
        assert "area_b" in entry

    def test_strengthening_detected(self):
        """After projection, weights should be strengthened."""
        lb = LiveBrain(p=0.1, seed=42)
        lb.add_area("A", n=100, k=10)
        lb.add_area("B", n=100, k=10)
        lb.add_stimulus("s", size=10)
        lb.stimulate("s", "A", rounds=10)
        lb.project("A", "B", rounds=10)
        summary = lb.get_area_edge_summary()
        entry = summary["A__B"]
        assert entry["max_weight"] > 1.0  # weights strengthened by Hebbian learning

    def test_undirected_key_order(self):
        """Pair key should always be sorted alphabetically."""
        lb = LiveBrain(p=0.1, seed=42)
        lb.add_area("Z", n=100, k=10)
        lb.add_area("A", n=100, k=10)
        summary = lb.get_area_edge_summary()
        assert "A__Z" in summary
        assert "Z__A" not in summary
