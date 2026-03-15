"""Tests for Flask API routes in app.py."""

import os
import sys
import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from viz.brain_api import LiveBrain
from viz.app import create_app


@pytest.fixture
def client():
    """Create a test client with a pre-configured brain."""
    brain = LiveBrain(p=0.1, seed=42)
    brain.add_area("A", n=100, k=10)
    brain.add_area("B", n=100, k=10)
    brain.add_stimulus("s", size=10)
    brain.stimulate("s", "A", rounds=5)
    brain.project("A", "B", rounds=3)
    app = create_app(brain)
    app.config["TESTING"] = True
    with app.server.test_client() as c:
        yield c


class TestAreaAPI:
    def test_get_area_data(self, client):
        resp = client.get("/api/area/A")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "nodes" in data
        assert "edges" in data
        assert "meta" in data
        assert len(data["nodes"]) == 100

    def test_area_meta(self, client):
        resp = client.get("/api/area/A")
        data = json.loads(resp.data)
        meta = data["meta"]
        assert meta["area"] == "A"
        assert meta["n"] == 100
        assert meta["n_winners"] == 10

    def test_nonexistent_area(self, client):
        resp = client.get("/api/area/NOPE")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["nodes"] == []
        assert data["meta"] == {}


class TestCrossAreaAPI:
    def test_get_cross_data(self, client):
        resp = client.get("/api/cross/A/B")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "nodes_a" in data
        assert "nodes_b" in data
        assert "edges_ab" in data
        assert "edges_ba" in data
        assert "meta" in data

    def test_cross_meta(self, client):
        resp = client.get("/api/cross/A/B")
        data = json.loads(resp.data)
        meta = data["meta"]
        assert meta["area_a"] == "A"
        assert meta["area_b"] == "B"

    def test_cross_has_edges(self, client):
        resp = client.get("/api/cross/A/B")
        data = json.loads(resp.data)
        # After projection A→B, should have A→B edges
        assert len(data["edges_ab"]) > 0

    def test_nonexistent_cross(self, client):
        resp = client.get("/api/cross/A/NOPE")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["nodes_a"] == []


class TestStaticPages:
    def test_area_detail_page(self, client):
        resp = client.get("/viz/area")
        assert resp.status_code == 200
        assert b"area_detail" in resp.data or b"Area Detail" in resp.data

    def test_cross_area_page(self, client):
        resp = client.get("/viz/cross")
        assert resp.status_code == 200
        assert b"cross_area" in resp.data or b"Cross-Area" in resp.data


class TestUndirectedGraph:
    """Test that the area graph now uses undirected edges."""

    def test_area_edge_summary_used(self, client):
        """The render_graph function should use get_area_edge_summary."""
        from viz.app import _render_graph, _brain
        graph = _render_graph(["A", "B"])
        # Should render without error and produce a Cytoscape component
        assert graph is not None

    def test_single_edge_per_pair(self):
        """Two areas should produce exactly one edge, not two."""
        brain = LiveBrain(p=0.1, seed=42)
        brain.add_area("X", n=50, k=5)
        brain.add_area("Y", n=50, k=5)
        create_app(brain)
        from viz.app import _render_graph
        import dash_cytoscape as cyto
        result = _render_graph(["X", "Y"])
        # Extract elements from the Cytoscape component
        elements = result.elements
        edges = [e for e in elements if "source" in e.get("data", {})]
        assert len(edges) == 1  # one undirected edge, not two directed
