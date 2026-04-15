"""
DFA Playground — Interactive page for building, training, and testing DFAs
on the Assembly Calculus model.

The main graph shows the BRAIN structure (STATE, INPUT, EDGES, RESULT areas)
with connectome edges — same style as the main workbench. Clicking an area
shows a neuron grid with winner/assembly info.
"""

import os
import sys
import math
import numpy as np

from dash import html, dcc, Input, Output, State, callback_context
import dash
import plotly.graph_objects as go
import dash_cytoscape as cyto

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dfa_sim import DFABrain, build_from_regex
from regex_to_dfa import regex_to_dfa, ParseError

# ── Module state ─────────────────────────────────────────────────────────────

_dfa_brain = None       # type: DFABrain | None
_dfa_spec = None        # type: dict | None

# ── Style constants (matching main app) ──────────────────────────────────────

DARK_BG = "#0f0f1a"
PANEL_BG = "#1a1a2e"
ACCENT = "#6366f1"
TEXT_COLOR = "#e0e0ee"
MUTED = "#888"
CHART_LAYOUT = dict(
    plot_bgcolor="#1a1a2e",
    paper_bgcolor="#1a1a2e",
    font={"color": "#aaa", "size": 10},
    margin={"l": 35, "r": 15, "t": 25, "b": 30},
)


def _panel(title, children, style_override=None):
    base = {
        "backgroundColor": PANEL_BG,
        "borderRadius": "12px",
        "padding": "16px",
        "border": "1px solid #2a2a4a",
        "marginBottom": "12px",
    }
    if style_override:
        base.update(style_override)
    return html.Div([
        html.H3(title, style={
            "color": TEXT_COLOR, "margin": "0 0 12px 0",
            "fontSize": "15px", "fontWeight": "600",
        }),
        html.Div(children),
    ], style=base)


def _btn(label, id, color="#2a2a4a", **kwargs):
    return html.Button(label, id=id, n_clicks=0, style={
        "backgroundColor": color, "color": TEXT_COLOR,
        "border": "1px solid #3a3a5a", "borderRadius": "6px",
        "padding": "6px 14px", "cursor": "pointer",
        "fontSize": "12px", "fontWeight": "500",
        **kwargs,
    })


def _input(id, placeholder="", type="text", value=None, width="100px"):
    return dcc.Input(
        id=id, type=type, placeholder=placeholder, value=value,
        style={
            "backgroundColor": "#1e1e3a", "color": TEXT_COLOR,
            "border": "1px solid #3a3a5a", "borderRadius": "4px",
            "padding": "5px 8px", "fontSize": "12px", "width": width,
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Layout
# ══════════════════════════════════════════════════════════════════════════════

def dfa_layout():
    """Return the Dash layout for the DFA Playground page."""
    return html.Div(
        style={
            "backgroundColor": DARK_BG, "minHeight": "100vh",
            "padding": "20px",
            "fontFamily": "'Inter', 'Segoe UI', sans-serif",
            "color": TEXT_COLOR,
        },
        children=[
            # Header
            html.Div(
                style={"display": "flex", "alignItems": "center",
                       "justifyContent": "space-between", "marginBottom": "20px"},
                children=[
                    html.H1("DFA Playground", style={
                        "margin": "0", "fontSize": "24px",
                        "background": f"linear-gradient(135deg, {ACCENT}, #a78bfa)",
                        "WebkitBackgroundClip": "text",
                        "WebkitTextFillColor": "transparent",
                    }),
                    dcc.Link(
                        html.Button("Back to Workbench", style={
                            "backgroundColor": "#2a2a4a", "color": TEXT_COLOR,
                            "border": "1px solid #3a3a5a", "borderRadius": "6px",
                            "padding": "6px 14px", "cursor": "pointer",
                            "fontSize": "12px",
                        }),
                        href="/",
                    ),
                ],
            ),

            # Main grid: sidebar + content
            html.Div(
                style={"display": "grid",
                       "gridTemplateColumns": "320px 1fr",
                       "gap": "16px"},
                children=[
                    # ── LEFT SIDEBAR ─────────────────────────────────────
                    html.Div([
                        # 1. Define DFA
                        _panel("Define DFA", [
                            html.Div("Regex pattern:", style={
                                "color": MUTED, "fontSize": "12px",
                                "marginBottom": "4px"}),
                            _input("dfa-regex", "(ab)*", width="260px"),
                            html.Div("Alphabet (comma-separated):", style={
                                "color": MUTED, "fontSize": "12px",
                                "marginTop": "8px", "marginBottom": "4px"}),
                            _input("dfa-alphabet", "a,b", width="260px"),
                            html.Div(style={"display": "flex", "gap": "6px",
                                            "marginTop": "8px",
                                            "alignItems": "center"},
                                     children=[
                                html.Div("k:", style={
                                    "color": MUTED, "fontSize": "12px"}),
                                _input("dfa-k", "", type="number",
                                       value=100, width="60px"),
                                html.Div("beta:", style={
                                    "color": MUTED, "fontSize": "12px"}),
                                _input("dfa-beta", "", type="number",
                                       value=0.05, width="60px"),
                            ]),
                            html.Div(style={"marginTop": "8px"}, children=[
                                _btn("Build DFA", "dfa-btn-build",
                                     color="#2a4a2a"),
                            ]),
                            html.Div(id="dfa-build-msg", style={
                                "fontSize": "11px", "marginTop": "6px",
                                "color": "#4ade80"}),
                        ]),

                        # 2. Training
                        _panel("Training", [
                            html.Div(style={"display": "flex", "gap": "6px",
                                            "alignItems": "center"},
                                     children=[
                                html.Span("Rounds:", style={
                                    "color": MUTED, "fontSize": "12px"}),
                                _input("dfa-rounds", "", type="number",
                                       value=30, width="60px"),
                                _btn("Train", "dfa-btn-train",
                                     color="#4a3a2a"),
                            ]),
                            html.Div(id="dfa-train-msg", style={
                                "fontSize": "11px", "marginTop": "6px",
                                "color": "#4ade80"}),
                        ]),

                        # 3. Test String
                        _panel("Test String", [
                            html.Div(style={"display": "flex", "gap": "6px",
                                            "alignItems": "center"},
                                     children=[
                                _input("dfa-test-input", "ab",
                                       width="180px"),
                                _btn("Run", "dfa-btn-test",
                                     color="#3a3a5a"),
                            ]),
                            html.Div(id="dfa-test-result", style={
                                "marginTop": "8px"}),
                        ]),

                        # 4. DFA Spec
                        _panel("DFA Specification", [
                            html.Div(id="dfa-spec-table", style={
                                "maxHeight": "200px", "overflowY": "auto",
                                "fontSize": "11px"}),
                        ]),

                        # 5. Training Log
                        _panel("Training Log", [
                            html.Div(id="dfa-train-log", style={
                                "maxHeight": "150px", "overflowY": "auto",
                                "fontSize": "11px", "fontFamily": "monospace"}),
                        ]),
                    ]),

                    # ── RIGHT: Visualization ─────────────────────────────
                    html.Div([
                        # Row 1: Brain Graph + Area Table
                        html.Div(
                            style={"display": "grid",
                                   "gridTemplateColumns": "1fr 1fr",
                                   "gap": "12px"},
                            children=[
                                _panel(
                                    "Brain Graph -- click area node to inspect",
                                    [
                                        html.Div(id="dfa-graph-container",
                                                 style={"minHeight": "320px"}),
                                        html.Div(id="dfa-edge-detail",
                                                 style={"marginTop": "8px"}),
                                        html.Div(id="dfa-neuron-detail",
                                                 style={"marginTop": "8px"}),
                                    ]),
                                _panel("Area Status", [
                                    html.Div(id="dfa-area-table",
                                             style={"maxHeight": "340px",
                                                    "overflowY": "auto"}),
                                ]),
                            ],
                        ),

                        # Row 2: Step-Through Debugger (full width)
                        _panel("Step-Through Debugger", [
                            html.Div(
                                style={"display": "flex", "gap": "12px",
                                       "alignItems": "center",
                                       "flexWrap": "wrap"},
                                children=[
                                    _input("dfa-step-input", "abab",
                                           width="140px"),
                                    _btn("Step Through", "dfa-btn-step",
                                         color="#2a3a5a"),
                                    _btn("◀ Prev", "dfa-btn-prev",
                                         color="#3a3a5a"),
                                    html.Div(id="dfa-step-counter", style={
                                        "fontSize": "13px", "color": MUTED,
                                        "minWidth": "50px",
                                        "textAlign": "center",
                                        "fontWeight": "600"}),
                                    _btn("Next ▶", "dfa-btn-next",
                                         color="#3a3a5a"),
                                    html.Div(id="dfa-step-error", style={
                                        "fontSize": "11px",
                                        "color": "#f87171",
                                        "marginLeft": "8px"}),
                                ],
                            ),
                            html.Div(id="dfa-step-display", style={
                                "marginTop": "8px"}),
                        ]),

                        # Row 2: Training Progress + Weight Histogram
                        html.Div(
                            style={"display": "grid",
                                   "gridTemplateColumns": "1fr 1fr",
                                   "gap": "12px"},
                            children=[
                                _panel("Training Progress", [
                                    dcc.Graph(id="dfa-training-chart",
                                              config={"displayModeBar": False},
                                              style={"height": "280px"}),
                                ]),
                                _panel("Weight Distribution", [
                                    html.Div(
                                        style={"display": "flex", "gap": "6px",
                                               "marginBottom": "6px"},
                                        children=[
                                            dcc.Dropdown(
                                                id="dfa-wt-from",
                                                placeholder="From",
                                                style={"width": "120px",
                                                       "backgroundColor": "#1e1e3a",
                                                       "color": "#000",
                                                       "fontSize": "12px"},
                                            ),
                                            html.Span("->", style={
                                                "color": MUTED,
                                                "lineHeight": "36px"}),
                                            dcc.Dropdown(
                                                id="dfa-wt-to",
                                                placeholder="To",
                                                style={"width": "120px",
                                                       "backgroundColor": "#1e1e3a",
                                                       "color": "#000",
                                                       "fontSize": "12px"},
                                            ),
                                        ],
                                    ),
                                    dcc.Graph(id="dfa-weight-hist",
                                              config={"displayModeBar": False},
                                              style={"height": "240px"}),
                                ]),
                            ],
                        ),
                    ]),
                ],
            ),

            # Hidden stores
            dcc.Store(id="dfa-refresh", data=0),
            dcc.Store(id="dfa-selected-area", data=None),
            dcc.Store(id="dfa-step-snapshots", data=None),
            dcc.Store(id="dfa-step-index", data=0),
        ],
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Callbacks
# ══════════════════════════════════════════════════════════════════════════════

def register_dfa_callbacks(app):
    """Register all DFA Playground callbacks on the given Dash app."""

    # == Build DFA ==
    @app.callback(
        Output("dfa-build-msg", "children"),
        Output("dfa-refresh", "data", allow_duplicate=True),
        Input("dfa-btn-build", "n_clicks"),
        State("dfa-regex", "value"),
        State("dfa-alphabet", "value"),
        State("dfa-k", "value"),
        State("dfa-beta", "value"),
        State("dfa-refresh", "data"),
        prevent_initial_call=True,
    )
    def build_dfa(n, regex, alpha_str, k, beta, refresh):
        global _dfa_brain, _dfa_spec
        if not regex:
            return "Enter a regex pattern.", refresh
        if not alpha_str:
            return "Enter an alphabet.", refresh

        try:
            alphabet = [c.strip() for c in alpha_str.split(",") if c.strip()]
            if not alphabet:
                return "Alphabet is empty.", refresh

            k = int(k or 100)
            beta = float(beta or 0.05)

            states, alpha, start, accept, trans = regex_to_dfa(
                regex, alphabet)

            _dfa_brain = DFABrain(
                states=states, alphabet=alpha,
                start_state=start, accept_states=accept,
                transitions=trans, k=k, beta=beta, seed=42,
            )
            _dfa_spec = {
                "regex": regex,
                "states": states,
                "alphabet": alpha,
                "start": start,
                "accept": accept,
                "transitions": trans,
            }
            n_states = len(states)
            n_trans = len(trans)
            return (f"Built DFA: {n_states} states, "
                    f"{n_trans} transitions, "
                    f"alphabet={alpha}",
                    refresh + 1)
        except ParseError as e:
            return html.Span(f"Regex error: {e}",
                             style={"color": "#f87171"}), refresh
        except Exception as e:
            return html.Span(f"Error: {e}",
                             style={"color": "#f87171"}), refresh

    # == Train ==
    @app.callback(
        Output("dfa-train-msg", "children"),
        Output("dfa-refresh", "data", allow_duplicate=True),
        Input("dfa-btn-train", "n_clicks"),
        State("dfa-rounds", "value"),
        State("dfa-refresh", "data"),
        prevent_initial_call=True,
    )
    def train_dfa(n, rounds, refresh):
        global _dfa_brain
        if _dfa_brain is None:
            return "Build a DFA first.", refresh
        rounds = int(rounds or 30)
        try:
            _dfa_brain.train(rounds=rounds, record=True)
            n_snaps = len(_dfa_brain.training_log)
            return (f"Training complete. {rounds} rounds, "
                    f"{n_snaps} snapshots recorded.",
                    refresh + 1)
        except Exception as e:
            return f"Training error: {e}", refresh

    # == Test String ==
    @app.callback(
        Output("dfa-test-result", "children"),
        Output("dfa-refresh", "data", allow_duplicate=True),
        Input("dfa-btn-test", "n_clicks"),
        State("dfa-test-input", "value"),
        State("dfa-refresh", "data"),
        prevent_initial_call=True,
    )
    def test_string(n, test_str, refresh):
        if _dfa_brain is None:
            return "Build and train a DFA first.", refresh
        if not _dfa_brain._trained:
            return "Train the DFA first.", refresh
        if test_str is None:
            test_str = ""
        try:
            for ch in test_str:
                if ch not in _dfa_brain.symbol_index:
                    return html.Div([
                        html.Span("Error: ", style={"color": "#f87171",
                                                     "fontWeight": "700"}),
                        html.Span(
                            f"Character '{ch}' is not in alphabet "
                            f"{_dfa_brain.alphabet}",
                            style={"color": "#f87171"}),
                    ]), refresh

            accepted, final, trace = _dfa_brain.run(test_str)
            display = test_str if test_str else "<empty>"
            icon = "ACCEPT" if accepted else "REJECT"
            color = "#4ade80" if accepted else "#f87171"

            return html.Div([
                html.Div([
                    html.Span(f"{icon} ", style={
                        "color": color, "fontWeight": "700",
                        "fontSize": "16px"}),
                    html.Span(f'"{display}"', style={
                        "color": TEXT_COLOR, "fontSize": "14px"}),
                ], style={"marginBottom": "6px"}),
                html.Div([
                    html.Span("Trace: ", style={
                        "color": MUTED, "fontSize": "12px"}),
                    html.Span(" -> ".join(trace), style={
                        "color": TEXT_COLOR, "fontSize": "12px",
                        "fontFamily": "monospace"}),
                ]),
            ]), refresh + 1
        except Exception as e:
            return f"Error: {e}", refresh

    # == Master refresh — updates all panels ==
    @app.callback(
        Output("dfa-graph-container", "children"),
        Output("dfa-area-table", "children"),
        Output("dfa-spec-table", "children"),
        Output("dfa-training-chart", "figure"),
        Output("dfa-train-log", "children"),
        Output("dfa-wt-from", "options"),
        Output("dfa-wt-to", "options"),
        Input("dfa-refresh", "data"),
    )
    def refresh_dfa(trigger):
        graph = _render_brain_graph()
        areas = _render_area_table()
        spec = _render_spec_table()
        train_chart = _render_training_chart()
        log = _render_train_log()
        area_opts = []
        if _dfa_brain:
            area_opts = [{"label": a, "value": a}
                         for a in _dfa_brain.area_by_name.keys()]
        return graph, areas, spec, train_chart, log, area_opts, area_opts

    # == Weight histogram (dropdown-driven) ==
    @app.callback(
        Output("dfa-weight-hist", "figure"),
        Input("dfa-wt-from", "value"),
        Input("dfa-wt-to", "value"),
        Input("dfa-refresh", "data"),
    )
    def update_weight_hist(from_area, to_area, trigger):
        return _render_weight_hist(from_area, to_area)

    # == Brain graph node click → select area ==
    @app.callback(
        Output("dfa-selected-area", "data"),
        Input("dfa-cyto-graph", "tapNodeData"),
        prevent_initial_call=True,
    )
    def on_node_click(node_data):
        if node_data and node_data.get("id"):
            return node_data["id"]
        raise dash.exceptions.PreventUpdate

    # == Neuron detail for selected area (uses snapshot if stepping) ==
    @app.callback(
        Output("dfa-neuron-detail", "children"),
        Input("dfa-selected-area", "data"),
        Input("dfa-refresh", "data"),
        Input("dfa-step-snapshots", "data"),
        Input("dfa-step-index", "data"),
    )
    def update_neuron_detail(area_name, trigger, snapshots, step_idx):
        if snapshots and step_idx is not None and 0 <= step_idx < len(snapshots):
            return _render_neuron_detail(area_name,
                                         snapshot=snapshots[step_idx])
        return _render_neuron_detail(area_name)

    # == Edge click -> show weight stats ==
    @app.callback(
        Output("dfa-edge-detail", "children"),
        Input("dfa-cyto-graph", "tapEdgeData"),
        prevent_initial_call=True,
    )
    def on_edge_click(edge_data):
        if edge_data and edge_data.get("area_a") and edge_data.get("area_b"):
            return _render_edge_weight_detail(
                edge_data["area_a"], edge_data["area_b"])
        raise dash.exceptions.PreventUpdate

    # ── Step-Through Debugger Callbacks ──────────────────────────────

    # == Start step-through ==
    @app.callback(
        Output("dfa-step-snapshots", "data"),
        Output("dfa-step-index", "data", allow_duplicate=True),
        Output("dfa-step-error", "children"),
        Input("dfa-btn-step", "n_clicks"),
        State("dfa-step-input", "value"),
        prevent_initial_call=True,
    )
    def start_stepping(n, step_str):
        if _dfa_brain is None:
            return None, 0, "Build and train a DFA first."
        if not _dfa_brain._trained:
            return None, 0, "Train the DFA first."
        if step_str is None:
            step_str = ""

        # Validate characters
        for ch in step_str:
            if ch not in _dfa_brain.symbol_index:
                return (None, 0,
                        f"Character '{ch}' not in alphabet "
                        f"{_dfa_brain.alphabet}")

        try:
            snapshots = _dfa_brain.run_with_snapshots(step_str)
            return snapshots, 0, ""
        except Exception as e:
            return None, 0, f"Error: {e}"

    # == Prev button ==
    @app.callback(
        Output("dfa-step-index", "data", allow_duplicate=True),
        Input("dfa-btn-prev", "n_clicks"),
        State("dfa-step-index", "data"),
        State("dfa-step-snapshots", "data"),
        prevent_initial_call=True,
    )
    def step_prev(n, idx, snapshots):
        if not snapshots:
            raise dash.exceptions.PreventUpdate
        new_idx = max(0, (idx or 0) - 1)
        return new_idx

    # == Next button ==
    @app.callback(
        Output("dfa-step-index", "data", allow_duplicate=True),
        Input("dfa-btn-next", "n_clicks"),
        State("dfa-step-index", "data"),
        State("dfa-step-snapshots", "data"),
        prevent_initial_call=True,
    )
    def step_next(n, idx, snapshots):
        if not snapshots:
            raise dash.exceptions.PreventUpdate
        new_idx = min(len(snapshots) - 1, (idx or 0) + 1)
        return new_idx

    # == Update step display, counter, and graph from snapshot ==
    @app.callback(
        Output("dfa-step-display", "children"),
        Output("dfa-step-counter", "children"),
        Output("dfa-graph-container", "children", allow_duplicate=True),
        Output("dfa-area-table", "children", allow_duplicate=True),
        Input("dfa-step-snapshots", "data"),
        Input("dfa-step-index", "data"),
        prevent_initial_call=True,
    )
    def update_step_view(snapshots, step_idx):
        if not snapshots or step_idx is None:
            return "", "", dash.no_update, dash.no_update

        if step_idx < 0 or step_idx >= len(snapshots):
            step_idx = 0

        snap = snapshots[step_idx]
        total = len(snapshots)

        # Build the step display
        display = _render_step_display(snap, snapshots)
        counter = f"{step_idx}/{total - 1}"
        graph = _render_brain_graph(snapshot=snap)
        table = _render_area_table(snapshot=snap)

        return display, counter, graph, table

    # ── Flask API Routes for DFA Sigma.js pages ──────────────────────

    from flask import jsonify

    @app.server.route("/api/dfa/area/<area_name>")
    def api_dfa_area(area_name):
        """Neuron-level graph data for the DFA brain (for Sigma.js)."""
        if _dfa_brain is None:
            return jsonify({"nodes": [], "edges": [], "meta": {}})
        return jsonify(_get_neuron_graph_data(area_name))

    @app.server.route("/api/dfa/cross/<area_a>/<area_b>")
    def api_dfa_cross(area_a, area_b):
        """Cross-area graph data for the DFA brain (for Sigma.js)."""
        if _dfa_brain is None:
            return jsonify({"nodes_a": [], "nodes_b": [],
                            "edges_ab": [], "edges_ba": [],
                            "meta": {}})
        return jsonify(_get_cross_area_graph_data(area_a, area_b))

# ══════════════════════════════════════════════════════════════════════════════
#  Sigma.js API data helpers
# ══════════════════════════════════════════════════════════════════════════════

def _get_neuron_graph_data(area_name):
    """Get neuron-level graph data for a DFA brain area (for Sigma.js)."""
    brain = _dfa_brain
    area = brain.area_by_name.get(area_name)
    if not area:
        return {"nodes": [], "edges": [], "meta": {}}

    winners = set(area.winners) if area.winners else set()

    # Build nodes
    nodes = []
    for i in range(area.n):
        node = {"id": i}
        if i in winners:
            node["state"] = "winner"
        else:
            node["state"] = "inactive"
        nodes.append(node)

    # Build edges from self-connectome
    edges = []
    conn = brain.connectomes.get(area_name, {}).get(area_name)
    if conn is not None and conn.size > 0 and winners:
        winner_list = sorted(winners)
        for src in winner_list:
            if src >= conn.shape[0]:
                continue
            row = conn[src]
            for tgt in range(min(area.n, conn.shape[1])):
                w = float(row[tgt])
                if w > 0 and src != tgt:
                    edges.append({
                        "source": src, "target": tgt,
                        "weight": round(w, 4),
                    })

    meta = {
        "area": area_name,
        "n": area.n,
        "k": area.k,
        "beta": area.beta,
        "n_winners": len(winners),
        "overlap": 0,
        "gained": 0,
        "lost": 0,
    }

    return {"nodes": nodes, "edges": edges, "meta": meta}


def _get_cross_area_graph_data(area_a, area_b):
    """Get bipartite graph data for two DFA brain areas (for Sigma.js)."""
    brain = _dfa_brain
    a_obj = brain.area_by_name.get(area_a)
    b_obj = brain.area_by_name.get(area_b)
    if not a_obj or not b_obj:
        return {"nodes_a": [], "nodes_b": [],
                "edges_ab": [], "edges_ba": [], "meta": {}}

    winners_a = set(a_obj.winners) if a_obj.winners else set()
    winners_b = set(b_obj.winners) if b_obj.winners else set()

    def _build_nodes(area_obj, winners):
        nodes = []
        for i in range(area_obj.n):
            state = "winner" if i in winners else "inactive"
            nodes.append({"id": i, "state": state})
        return nodes

    def _build_edges(from_name, to_name, from_winners):
        edges = []
        conn = brain.connectomes.get(from_name, {}).get(to_name)
        if conn is None or conn.size == 0:
            return edges
        for src in sorted(from_winners):
            if src >= conn.shape[0]:
                continue
            row = conn[src]
            for tgt in range(conn.shape[1]):
                w = float(row[tgt])
                if w > 0:
                    edges.append({
                        "source": src, "target": tgt,
                        "weight": round(w, 4),
                    })
        return edges

    nodes_a = _build_nodes(a_obj, winners_a)
    nodes_b = _build_nodes(b_obj, winners_b)
    edges_ab = _build_edges(area_a, area_b, winners_a)
    edges_ba = _build_edges(area_b, area_a, winners_b)

    meta = {
        "n_a": a_obj.n, "n_b": b_obj.n,
        "winners_a": len(winners_a), "winners_b": len(winners_b),
        "edges_ab_count": len(edges_ab),
        "edges_ba_count": len(edges_ba),
    }

    return {"nodes_a": nodes_a, "nodes_b": nodes_b,
            "edges_ab": edges_ab, "edges_ba": edges_ba,
            "meta": meta}


# ══════════════════════════════════════════════════════════════════════════════
#  Render helpers
# ══════════════════════════════════════════════════════════════════════════════

def _render_brain_graph(snapshot=None):
    """Render the brain areas (STATE, INPUT, EDGES, RESULT) as a Cytoscape
    graph with connectome edges. If snapshot is provided, uses its winner data."""
    if _dfa_brain is None:
        return html.Div("Build a DFA to see the brain graph.",
                        style={"color": "#666", "fontSize": "13px",
                               "padding": "40px", "textAlign": "center"})

    brain = _dfa_brain
    areas = list(brain.area_by_name.keys())

    elements = []
    n_areas = len(areas)

    # Arrange areas in a circle
    for i, area_name in enumerate(areas):
        angle = 2 * math.pi * i / max(n_areas, 1) - math.pi / 2
        x = 200 + 130 * math.cos(angle)
        y = 200 + 130 * math.sin(angle)

        area = brain.area_by_name[area_name]

        # Use snapshot data if available
        if snapshot and "areas" in snapshot:
            snap_area = snapshot["areas"].get(area_name, {})
            n_winners = len(snap_area.get("winners", []))
            asm_label = snap_area.get("assembly_label", "?")
            asm_pct = snap_area.get("assembly_pct", 0)
        else:
            winners = area.winners if area.winners else []
            n_winners = len(winners)
            asm_label = None
            asm_pct = None

        classes = ["area"]
        if n_winners > 0:
            classes.append("active")

        label = f"{area_name}\n{area.n}n, k={area.k}"
        if n_winners > 0:
            label += f"\n{n_winners}w"
        if asm_label and asm_pct and asm_pct > 0.5:
            label += f"\n[{asm_label}]"

        elements.append({
            "data": {"id": area_name, "label": label},
            "position": {"x": x, "y": y},
            "classes": " ".join(classes),
        })

    # Add edges between areas (undirected, one per pair)
    seen = set()
    for a in areas:
        for b in areas:
            if a == b:
                continue
            pair = "__".join(sorted([a, b]))
            if pair in seen:
                continue
            seen.add(pair)

            # Check if connectome exists between these areas
            conn_ab = brain.connectomes.get(a, {}).get(b)
            conn_ba = brain.connectomes.get(b, {}).get(a)
            has_conn = (conn_ab is not None and conn_ab.size > 0) or \
                       (conn_ba is not None and conn_ba.size > 0)

            if not has_conn:
                continue

            # Weight stats for edge styling
            max_w = 0
            pct_strong = 0
            for conn in [conn_ab, conn_ba]:
                if conn is not None and conn.size > 0:
                    nz = conn[conn > 0]
                    if len(nz) > 0:
                        max_w = max(max_w, float(np.max(nz)))
                        strong = nz[nz > 1.001]
                        pct_strong = max(pct_strong,
                                         len(strong) / len(nz) * 100
                                         if len(nz) > 0 else 0)

            classes = []
            if max_w > 1.1:
                classes.append("strengthened")

            edge_label = ""
            if pct_strong > 0:
                edge_label = f"+{pct_strong:.0f}%"

            elements.append({
                "data": {
                    "source": a, "target": b,
                    "label": edge_label,
                    "pair_key": pair,
                    "area_a": sorted([a, b])[0],
                    "area_b": sorted([a, b])[1],
                },
                "classes": " ".join(classes),
            })

    # Stylesheet matching main app
    stylesheet = [
        {"selector": "node",
         "style": {"label": "data(label)",
                   "text-valign": "center", "text-halign": "center",
                   "background-color": "#2a2a4a",
                   "color": "#e0e0ee", "font-size": "11px",
                   "border-width": 2, "border-color": "#3a3a5a",
                   "width": 80, "height": 80,
                   "text-wrap": "wrap", "text-max-width": "75px"}},
        {"selector": "node.active",
         "style": {"background-color": "#6366f1",
                   "border-color": "#a78bfa",
                   "border-width": 3, "font-weight": "bold"}},
        {"selector": "edge",
         "style": {"width": 1, "line-color": "#3a3a5a",
                   "curve-style": "bezier",
                   "target-arrow-shape": "none",
                   "label": "data(label)",
                   "font-size": "8px", "color": "#888",
                   "text-rotation": "autorotate"}},
        {"selector": "edge.active",
         "style": {"line-color": "#4a4a6a", "width": 1.5}},
        {"selector": "edge.strengthened",
         "style": {"line-color": "#6366f1", "width": 2.5,
                   "color": "#a78bfa"}},
    ]

    return cyto.Cytoscape(
        id="dfa-cyto-graph",
        elements=elements,
        stylesheet=stylesheet,
        layout={"name": "preset"},
        style={"width": "100%", "height": "320px",
               "backgroundColor": "#0f0f1a"},
        userZoomingEnabled=True,
        userPanningEnabled=True,
    )


def _render_neuron_detail(area_name, snapshot=None):
    """Render neuron grid for the selected brain area.
    If snapshot is provided, uses its winner data instead of live brain."""
    if not area_name or _dfa_brain is None:
        return html.Div("Click an area node above to inspect neurons.",
                        style={"color": "#666", "fontSize": "12px",
                               "padding": "8px", "textAlign": "center"})

    area = _dfa_brain.area_by_name.get(area_name)
    if not area:
        return html.Div(f"Area '{area_name}' not found.",
                        style={"color": "#f87171", "fontSize": "12px"})

    # Use snapshot winners if available, otherwise live brain
    if snapshot and "areas" in snapshot:
        snap_area = snapshot["areas"].get(area_name, {})
        winners = set(snap_area.get("winners", []))
    else:
        winners = set(area.winners) if area.winners else set()
    n = area.n
    k = _dfa_brain.k_val

    # Build assembly labels for this area
    assembly_labels = _get_assembly_labels(area_name)

    # For each assembly, show which ones have winners
    assembly_info = []
    for idx, label in enumerate(assembly_labels):
        start = idx * k
        end = start + k
        asm_neurons = set(range(start, end))
        overlap = len(winners & asm_neurons)
        if overlap > 0:
            pct = overlap / k * 100
            assembly_info.append(
                html.Span(f" [{label}: {pct:.0f}%] ", style={
                    "color": "#4ade80" if pct > 50 else "#fbbf24",
                    "fontSize": "11px", "marginRight": "4px"}))

    # Summary header
    explore_btn = html.A(
        "Explore Neurons",
        href=f"/viz/area?area={area_name}&source=dfa",
        target="_blank",
        style={
            "backgroundColor": "#6366f1", "color": "#fff",
            "border": "none", "borderRadius": "6px",
            "padding": "4px 12px", "cursor": "pointer",
            "fontSize": "11px", "fontWeight": "600",
            "textDecoration": "none", "marginLeft": "auto",
        },
    )

    header = html.Div([
        html.Span(f"Area '{area_name}'", style={
            "fontWeight": "700", "fontSize": "14px", "marginRight": "12px"}),
        html.Span(f"{len(winners)} winners", style={
            "color": "#4ade80" if winners else "#666",
            "fontSize": "11px", "marginRight": "8px"}),
        html.Span(f"{n} neurons total", style={
            "color": "#666", "fontSize": "11px"}),
        explore_btn,
    ], style={"marginBottom": "4px", "display": "flex",
              "alignItems": "center", "flexWrap": "wrap", "gap": "4px"})

    # Assembly badges
    asm_row = html.Div(assembly_info, style={
        "marginBottom": "6px"}) if assembly_info else None

    # Neuron grid — show up to 500 neurons
    max_display = min(n, 500)
    dots = []
    for i in range(max_display):
        if i in winners:
            bg = "#4ade80"
            border = "1px solid #22c55e"
            # Find which assembly this neuron belongs to
            asm_idx = i // k
            asm_label = (assembly_labels[asm_idx]
                         if asm_idx < len(assembly_labels)
                         else f"asm{asm_idx}")
            title = f"Neuron {i} -- WINNER (assembly: {asm_label})"
        else:
            bg = "#2a2a3a"
            border = "1px solid #1a1a2a"
            title = f"Neuron {i} -- inactive"

        dots.append(html.Div(
            title=title,
            style={
                "width": "6px", "height": "6px",
                "backgroundColor": bg, "border": border,
                "borderRadius": "1px", "display": "inline-block",
            },
        ))

    suffix = ""
    if n > max_display:
        suffix = f"  ... showing {max_display}/{n} neurons"

    grid = html.Div(dots, style={
        "display": "flex", "flexWrap": "wrap", "gap": "2px"})

    footer = html.Div(suffix, style={
        "color": "#666", "fontSize": "10px", "marginTop": "4px",
    }) if suffix else None

    children = [header]
    if asm_row:
        children.append(asm_row)
    children.append(grid)
    if footer:
        children.append(footer)

    return html.Div(children, style={
        "backgroundColor": "#0f0f1a", "borderRadius": "8px",
        "padding": "10px", "border": "1px solid #2a2a4a",
    })


def _get_assembly_labels(area_name):
    """Get human-readable labels for each assembly in a brain area."""
    if _dfa_brain is None:
        return []

    k = _dfa_brain.k_val
    if area_name == "STATE" or area_name == "RESULT":
        return list(_dfa_brain.states)
    elif area_name == "INPUT":
        return list(_dfa_brain.alphabet)
    elif area_name == "EDGES":
        labels = []
        for s in _dfa_brain.states:
            for sym in _dfa_brain.alphabet:
                dst = _dfa_brain.transitions.get((s, sym), "?")
                labels.append(f"{s},{sym}->{dst}")
        return labels
    return []


def _render_edge_weight_detail(area_a, area_b):
    """Render weight stats for a clicked edge between two areas."""
    if _dfa_brain is None:
        return html.Div()

    brain = _dfa_brain

    def _get_stats(from_a, to_b):
        conn = brain.connectomes.get(from_a, {}).get(to_b)
        if conn is None or conn.size == 0:
            return {"mean_nonzero": 0, "max": 0, "nonzero": 0, "std": 0}
        nz = conn[conn > 0]
        if len(nz) == 0:
            return {"mean_nonzero": 0, "max": 0, "nonzero": 0, "std": 0}
        return {
            "mean_nonzero": float(np.mean(nz)),
            "max": float(np.max(nz)),
            "nonzero": int(len(nz)),
            "std": float(np.std(nz)),
        }

    stats_ab = _get_stats(area_a, area_b)
    stats_ba = _get_stats(area_b, area_a)

    def _stat_row(label, val_ab, val_ba, fmt=".4f", color="#ddd"):
        return html.Tr([
            html.Td(label, style={"padding": "3px 8px", "fontSize": "12px",
                                   "color": "#aaa", "fontWeight": "600"}),
            html.Td(f"{val_ab:{fmt}}", style={"padding": "3px 8px",
                     "fontSize": "12px", "color": color,
                     "textAlign": "right"}),
            html.Td(f"{val_ba:{fmt}}", style={"padding": "3px 8px",
                     "fontSize": "12px", "color": color,
                     "textAlign": "right"}),
        ])

    max_w = max(stats_ab.get("max", 0), stats_ba.get("max", 0))
    strengthened = max_w > 1.001
    badge_color = "#4ade80" if strengthened else "#666"
    badge_text = (f"Strengthened (max {max_w:.4f})"
                  if strengthened else "No strengthening yet")

    table = html.Table([
        html.Tr([
            html.Th("", style={"padding": "3px 8px"}),
            html.Th(f"{area_a} -> {area_b}", style={
                "padding": "3px 8px", "fontSize": "11px",
                "color": "#6366f1", "textAlign": "right"}),
            html.Th(f"{area_b} -> {area_a}", style={
                "padding": "3px 8px", "fontSize": "11px",
                "color": "#a78bfa", "textAlign": "right"}),
        ]),
        _stat_row("Mean (nonzero)",
                  stats_ab.get("mean_nonzero", 0),
                  stats_ba.get("mean_nonzero", 0)),
        _stat_row("Max weight",
                  stats_ab.get("max", 0), stats_ba.get("max", 0),
                  color="#4ade80" if strengthened else "#ddd"),
        _stat_row("Nonzero conns",
                  stats_ab.get("nonzero", 0), stats_ba.get("nonzero", 0),
                  fmt="d", color="#6366f1"),
        _stat_row("Std dev",
                  stats_ab.get("std", 0), stats_ba.get("std", 0)),
    ], style={"width": "100%", "borderCollapse": "collapse"})

    explore_btn = html.A(
        "Explore Connections",
        href=f"/viz/cross?a={area_a}&b={area_b}&source=dfa",
        target="_blank",
        style={
            "backgroundColor": "#6366f1", "color": "#fff",
            "border": "none", "borderRadius": "6px",
            "padding": "4px 12px", "cursor": "pointer",
            "fontSize": "11px", "fontWeight": "600",
            "textDecoration": "none", "marginLeft": "auto",
        },
    )

    return html.Div([
        html.Div([
            html.Span(f"Edge: {area_a} <-> {area_b}", style={
                "fontWeight": "700", "fontSize": "13px",
                "marginRight": "12px"}),
            html.Span(badge_text, style={
                "color": badge_color, "fontSize": "11px"}),
            explore_btn,
        ], style={"marginBottom": "6px", "display": "flex",
                  "alignItems": "center", "gap": "8px"}),
        table,
    ], style={
        "backgroundColor": "#0f0f1a", "borderRadius": "8px",
        "padding": "10px", "border": "1px solid #2a2a4a",
    })


def _render_area_table(snapshot=None):
    """Render area status table. If snapshot provided, uses its data."""
    if _dfa_brain is None:
        return html.Div("No DFA built yet.", style={"color": "#666"})

    brain = _dfa_brain
    k = brain.k_val

    row_style = {"padding": "3px 8px", "fontSize": "12px",
                 "borderBottom": "1px solid #1e1e3a", "color": "#ddd"}

    headers = ["Area", "n", "k", "Winners", "Assembly", "Confidence"]
    rows = [html.Tr([
        html.Th(h, style={"padding": "4px 8px",
                          "borderBottom": "1px solid #3a3a5a",
                          "color": "#aaa", "fontSize": "11px",
                          "textAlign": "left"})
        for h in headers
    ])]

    for name in ["STATE", "INPUT", "EDGES", "RESULT"]:
        area = brain.area_by_name.get(name)
        if not area:
            continue

        if snapshot and "areas" in snapshot:
            snap_area = snapshot["areas"].get(name, {})
            n_winners = len(snap_area.get("winners", []))
            asm_label = snap_area.get("assembly_label", "?")
            asm_pct = snap_area.get("assembly_pct", 0)
        else:
            winners = area.winners if area.winners else []
            n_winners = len(winners)
            asm_label = "-"
            asm_pct = 0

        pct_str = f"{asm_pct:.0%}" if asm_pct > 0 else "-"
        pct_color = "#4ade80" if asm_pct > 0.8 else (
            "#fbbf24" if asm_pct > 0.4 else "#666")

        rows.append(html.Tr([
            html.Td(name, style={**row_style, "fontWeight": "600",
                                 "color": ACCENT}),
            html.Td(str(area.n), style=row_style),
            html.Td(str(area.k), style=row_style),
            html.Td(str(n_winners), style={
                **row_style,
                "color": "#4ade80" if n_winners > 0 else "#666"}),
            html.Td(asm_label, style={
                **row_style, "fontWeight": "600",
                "color": "#a78bfa" if asm_pct > 0.5 else "#666"}),
            html.Td(pct_str, style={
                **row_style, "color": pct_color}),
        ]))

    return html.Table(rows, style={"width": "100%",
                                    "borderCollapse": "collapse"})


def _render_step_display(snap, all_snapshots):
    """Render the step-through debugger display for the current step."""
    step = snap["step"]
    symbol = snap.get("symbol")
    state_from = snap["state_from"]
    state_to = snap["state_to"]
    accepted = snap["accepted"]
    confidence = snap.get("confidence", 0)

    # Build the string visualization with current char highlighted
    # Reconstruct the original string from all snapshots
    chars = [s.get("symbol") or "" for s in all_snapshots]
    full_string = "".join(chars)

    if step == 0:
        # Step 0: initial state, show full string with no highlight
        string_parts = [
            html.Span("Initial state", style={
                "color": "#a78bfa", "fontWeight": "600",
                "fontSize": "13px"}),
        ]
        if full_string:
            string_parts.append(html.Span(
                f'  String: "{full_string}"',
                style={"color": "#666", "fontSize": "12px"}))
    else:
        # Steps 1+: highlight current character
        string_parts = []
        char_idx = step - 1  # 0-based index into the string
        for i, ch in enumerate(full_string):
            if i == char_idx:
                string_parts.append(html.Span(ch, style={
                    "color": "#fff", "backgroundColor": "#6366f1",
                    "padding": "1px 4px", "borderRadius": "3px",
                    "fontWeight": "700", "fontSize": "15px",
                    "fontFamily": "monospace"}))
            elif i < char_idx:
                # Already processed
                string_parts.append(html.Span(ch, style={
                    "color": "#4ade80", "fontSize": "14px",
                    "fontFamily": "monospace", "opacity": "0.7"}))
            else:
                # Not yet processed
                string_parts.append(html.Span(ch, style={
                    "color": "#666", "fontSize": "14px",
                    "fontFamily": "monospace"}))

    string_row = html.Div(string_parts, style={
        "marginBottom": "6px", "letterSpacing": "2px"})

    # State transition
    if step == 0:
        transition = html.Div([
            html.Span("State: ", style={"color": MUTED, "fontSize": "12px"}),
            html.Span(state_from, style={
                "color": "#4ade80" if accepted else "#ddd",
                "fontWeight": "700", "fontSize": "13px",
                "fontFamily": "monospace"}),
            html.Span(" (start)", style={
                "color": MUTED, "fontSize": "11px"}),
        ])
    else:
        transition = html.Div([
            html.Span(state_from, style={
                "color": "#ddd", "fontWeight": "600",
                "fontSize": "13px", "fontFamily": "monospace"}),
            html.Span(f" --({symbol})--> ", style={
                "color": "#a78bfa", "fontSize": "12px"}),
            html.Span(state_to, style={
                "color": "#4ade80" if accepted else "#f87171",
                "fontWeight": "700", "fontSize": "13px",
                "fontFamily": "monospace"}),
            html.Span(f"  conf: {confidence:.0%}", style={
                "color": MUTED, "fontSize": "11px",
                "marginLeft": "8px"}),
        ])

    # Accept/reject badge
    is_final = (step == len(all_snapshots) - 1)
    if is_final and step > 0:
        badge = html.Span(
            "ACCEPT" if accepted else "REJECT",
            style={
                "backgroundColor": "#166534" if accepted else "#7f1d1d",
                "color": "#4ade80" if accepted else "#f87171",
                "padding": "2px 8px", "borderRadius": "4px",
                "fontSize": "11px", "fontWeight": "700",
                "marginLeft": "8px",
            })
    else:
        badge_color = "#4ade80" if accepted else "#666"
        badge = html.Span(
            "accept state" if accepted else "",
            style={"color": badge_color, "fontSize": "10px",
                   "marginLeft": "8px"})

    return html.Div([
        string_row,
        html.Div([transition, badge], style={
            "display": "flex", "alignItems": "center"}),
    ], style={
        "backgroundColor": "#0f0f1a", "borderRadius": "6px",
        "padding": "8px", "border": "1px solid #2a2a4a",
    })


def _get_area_assembly_info():
    """Get assembly count and role description for each area."""
    if _dfa_brain is None:
        return {}
    ns = len(_dfa_brain.states)
    na = len(_dfa_brain.alphabet)
    return {
        "STATE": (f"{ns} states", "Current DFA state"),
        "INPUT": (f"{na} symbols", "Input symbol"),
        "EDGES": (f"{ns * na} pairs", "(state,symbol) pairs"),
        "RESULT": (f"{ns} states", "Next state readout"),
    }


def _render_spec_table():
    """Render the DFA transition table."""
    if _dfa_spec is None:
        return html.Div("Build a DFA to see its spec.",
                        style={"color": "#666", "fontSize": "12px"})

    transitions = _dfa_spec["transitions"]
    states = _dfa_spec["states"]
    alphabet = _dfa_spec["alphabet"]
    accept = set(_dfa_spec["accept"])
    start = _dfa_spec["start"]

    row_style = {"padding": "2px 6px", "fontSize": "11px",
                 "borderBottom": "1px solid #1e1e3a", "color": "#ddd"}
    header = [html.Th("State", style={**row_style, "color": "#aaa"})]
    for sym in alphabet:
        header.append(html.Th(sym, style={**row_style, "color": "#a78bfa"}))
    rows = [html.Tr(header)]

    for s in states:
        cells = []
        label = s
        if s == start:
            label = f"> {s}"
        if s in accept:
            label = f"* {s}"
        cells.append(html.Td(label, style={
            **row_style, "fontWeight": "600",
            "color": "#4ade80" if s in accept else TEXT_COLOR}))
        for sym in alphabet:
            dst = transitions.get((s, sym), "?")
            cells.append(html.Td(dst, style={
                **row_style,
                "color": "#666" if dst == "q_trash" else "#ddd"}))
        rows.append(html.Tr(cells))

    return html.Table(rows, style={"width": "100%",
                                    "borderCollapse": "collapse"})


def _render_training_chart():
    """Render training progress chart (max weight growth over steps)."""
    fig = go.Figure()

    if _dfa_brain and _dfa_brain.training_log:
        log = _dfa_brain.training_log
        steps = list(range(len(log)))

        for conn_name in ["STATE->EDGES", "INPUT->EDGES", "EDGES->RESULT"]:
            vals = [s["weights"].get(conn_name, {}).get("max", 0)
                    for s in log]
            fig.add_trace(go.Scatter(
                x=steps, y=vals,
                mode="lines",
                name=conn_name,
                line={"width": 2},
            ))

        # Mark phase boundary
        phase1_end = None
        for i, s in enumerate(log):
            if s["phase"] == "phase2" and (i == 0 or
                                            log[i-1]["phase"] == "phase1"):
                phase1_end = i
                break
        if phase1_end:
            fig.add_vline(x=phase1_end, line_dash="dot",
                         line_color="#fbbf24", opacity=0.7,
                         annotation_text="Phase 2 start",
                         annotation_font_size=9,
                         annotation_font_color="#fbbf24")

    fig.update_layout(
        **CHART_LAYOUT,
        xaxis={"gridcolor": "#2a2a4a", "title": "Training Step",
               "title_font_size": 10},
        yaxis={"gridcolor": "#2a2a4a", "title": "Max Weight",
               "title_font_size": 10},
        legend={"orientation": "h", "y": 1.15, "font": {"size": 9}},
        height=260,
    )
    return fig


def _render_weight_hist(from_area=None, to_area=None):
    """Render weight distribution histogram for a selected connectome pair."""
    fig = go.Figure()
    title_text = "Select areas above"

    if _dfa_brain and from_area and to_area:
        conn = _dfa_brain.connectomes.get(from_area, {}).get(to_area)
        if conn is not None and conn.size > 0:
            flat = conn.flatten()
            nonzero = flat[flat > 0]
            if len(nonzero) > 5000:
                idx = np.random.choice(len(nonzero), 5000, replace=False)
                nonzero = nonzero[idx]

            if len(nonzero) > 0:
                unlearned = [w for w in nonzero if w <= 1.001]
                strengthened = [w for w in nonzero if w > 1.001]
                max_w = float(np.max(nonzero))

                if unlearned:
                    fig.add_trace(go.Histogram(
                        x=unlearned, nbinsx=20,
                        name=f"Baseline ({len(unlearned)})",
                        marker={"color": "#4b5563",
                                "line": {"color": "#6b7280", "width": 0.5}},
                        opacity=0.7,
                    ))
                if strengthened:
                    fig.add_trace(go.Histogram(
                        x=strengthened, nbinsx=30,
                        name=f"Strengthened ({len(strengthened)})",
                        marker={"color": "#6366f1",
                                "line": {"color": "#a78bfa", "width": 0.5}},
                    ))

                pct = (len(strengthened) / len(nonzero) * 100
                       if nonzero.size > 0 else 0)
                title_text = (f"{from_area} -> {to_area}  |  "
                              f"{len(strengthened)}/{len(nonzero)} "
                              f"strengthened ({pct:.0f}%)  |  "
                              f"max: {max_w:.3f}")
            else:
                title_text = f"{from_area} -> {to_area}  |  No connections"
        else:
            title_text = f"{from_area} -> {to_area}  |  No weight data"

    fig.update_layout(
        **CHART_LAYOUT,
        xaxis={"gridcolor": "#2a2a4a", "title": "Weight",
               "title_font_size": 10},
        yaxis={"gridcolor": "#2a2a4a", "title": "Count",
               "title_font_size": 10},
        barmode="overlay",
        legend={"font": {"size": 9}, "x": 0.6, "y": 0.95,
                "bgcolor": "rgba(0,0,0,0.5)"},
        height=220,
        title={"text": title_text, "font": {"size": 10, "color": "#aaa"},
               "x": 0.5, "xanchor": "center"},
    )
    return fig


def _render_train_log():
    """Render the training readout results."""
    if _dfa_brain is None or not _dfa_brain.training_log:
        return html.Div("No training data.", style={"color": "#666"})

    log = _dfa_brain.training_log
    readout_entries = [s for s in log if s.get("readout")]
    if not readout_entries:
        return html.Div(f"{len(log)} training steps recorded.",
                        style={"color": MUTED})

    last = readout_entries[-1]
    items = []
    readout = last.get("readout", {})
    all_correct = True
    for key, info in sorted(readout.items()):
        correct = info["correct"]
        if not correct:
            all_correct = False
        icon = "OK" if correct else "FAIL"
        color = "#4ade80" if correct else "#f87171"
        items.append(html.Div(
            f"[{icon}] {key}  conf={info['confidence']:.2f}",
            style={"color": color, "padding": "1px 0"},
        ))

    summary_color = "#4ade80" if all_correct else "#f87171"
    n_correct = sum(1 for v in readout.values() if v["correct"])
    n_total = len(readout)
    items.insert(0, html.Div(
        f"Readout: {n_correct}/{n_total} transitions correct "
        f"({len(log)} total steps)",
        style={"color": summary_color, "fontWeight": "600",
               "marginBottom": "4px", "fontSize": "12px"},
    ))

    return html.Div(items)
