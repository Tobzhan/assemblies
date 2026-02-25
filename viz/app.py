"""
Assembly Calculus Workbench — Live Interactive Sandbox

An interactive dashboard where you create brain areas, add stimuli,
project, and watch assemblies form in real-time. Toggle learning ON/OFF
to probe the brain without changing weights.

Panels:
  Left sidebar: Controls (add area, stimulus, operations, learning toggle)
  Right: Live Cytoscape graph, convergence chart, area table, weight histogram
"""

import os
import sys
import math

from dash import Dash, html, dcc, Input, Output, State
import dash
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from viz.brain_api import LiveBrain
from viz.graph_builder import build_area_graph, create_graph_component

# ── Globals ──────────────────────────────────────────────────────────────────

_app = None
_brain = LiveBrain(p=0.1, beta=0.05, seed=42)


def create_app(brain=None):
    """Create and return a configured Dash app."""
    global _brain, _app
    if brain is not None:
        _brain = brain

    app = Dash(__name__)
    _app = app

    # ── Styles ───────────────────────────────────────────────────────────
    dark_bg = "#0f0f1a"
    panel_bg = "#1a1a2e"
    accent = "#6366f1"
    text_color = "#e0e0ee"
    muted = "#888"

    def _panel(title, children, style_override=None):
        base_style = {
            "backgroundColor": panel_bg,
            "borderRadius": "12px",
            "padding": "16px",
            "border": "1px solid #2a2a4a",
            "marginBottom": "12px",
        }
        if style_override:
            base_style.update(style_override)
        return html.Div([
            html.H3(title, style={
                "color": text_color, "margin": "0 0 12px 0",
                "fontSize": "15px", "fontWeight": "600",
            }),
            html.Div(children),
        ], style=base_style)

    def _btn(label, id, color="#2a2a4a", **kwargs):
        return html.Button(label, id=id, n_clicks=0, style={
            "backgroundColor": color, "color": text_color,
            "border": "1px solid #3a3a5a", "borderRadius": "6px",
            "padding": "6px 14px", "cursor": "pointer",
            "fontSize": "12px", "fontWeight": "500",
            **kwargs,
        })

    def _input(id, placeholder="", type="text", value=None, width="100px"):
        return dcc.Input(
            id=id, type=type, placeholder=placeholder, value=value,
            style={
                "backgroundColor": "#1e1e3a", "color": text_color,
                "border": "1px solid #3a3a5a", "borderRadius": "4px",
                "padding": "5px 8px", "fontSize": "12px", "width": width,
            },
        )

    def _dropdown(id, options=None, value=None, placeholder="", width="140px"):
        return dcc.Dropdown(
            id=id,
            options=options or [],
            value=value,
            placeholder=placeholder,
            style={"width": width, "backgroundColor": "#1e1e3a",
                   "color": "#000", "fontSize": "12px"},
        )

    # ── Layout ───────────────────────────────────────────────────────────

    app.layout = html.Div(
        style={
            "backgroundColor": dark_bg, "minHeight": "100vh",
            "padding": "20px", "fontFamily": "'Inter', 'Segoe UI', sans-serif",
            "color": text_color,
        },
        children=[
            # Header
            html.Div(
                style={"display": "flex", "alignItems": "center",
                       "justifyContent": "space-between", "marginBottom": "20px"},
                children=[
                    html.H1("🧠 Assembly Calculus Workbench", style={
                        "margin": "0", "fontSize": "24px",
                        "background": f"linear-gradient(135deg, {accent}, #a78bfa)",
                        "WebkitBackgroundClip": "text",
                        "WebkitTextFillColor": "transparent",
                    }),
                    html.Div(id="status-bar", style={
                        "fontSize": "13px", "color": muted,
                    }),
                ],
            ),

            # Main grid: sidebar + content
            html.Div(
                style={"display": "grid",
                       "gridTemplateColumns": "320px 1fr",
                       "gap": "16px"},
                children=[
                    # ── LEFT SIDEBAR: Controls ───────────────────────
                    html.Div([
                        # 1. Add Area
                        _panel("➕ Add Area", [
                            html.Div(style={"display": "flex", "gap": "6px",
                                            "flexWrap": "wrap", "alignItems": "center"},
                                     children=[
                                _input("area-name", "Name", width="70px"),
                                _input("area-n", "n", type="number", value=500, width="60px"),
                                _input("area-k", "k", type="number", value=50, width="55px"),
                                _input("area-beta", "β", type="number", value=0.05, width="55px"),
                                _btn("Add", "btn-add-area", color="#2a4a2a"),
                            ]),
                            html.Div(id="area-msg", style={"fontSize": "11px",
                                                            "color": "#4ade80", "marginTop": "4px"}),
                        ]),

                        # 2. Add Stimulus
                        _panel("⚡ Add Stimulus", [
                            html.Div(style={"display": "flex", "gap": "6px",
                                            "alignItems": "center"},
                                     children=[
                                _input("stim-name", "Name", width="80px"),
                                _input("stim-size", "Size", type="number", value=100, width="60px"),
                                _btn("Add", "btn-add-stim", color="#2a4a2a"),
                            ]),
                            html.Div(id="stim-msg", style={"fontSize": "11px",
                                                            "color": "#4ade80", "marginTop": "4px"}),
                        ]),

                        # 3. Operations
                        _panel("▶ Operations", [
                            # Stimulate
                            html.Div("Stimulate:", style={"color": muted,
                                        "fontSize": "12px", "marginBottom": "4px"}),
                            html.Div(style={"display": "flex", "gap": "6px",
                                            "alignItems": "center", "marginBottom": "10px"},
                                     children=[
                                _dropdown("op-stim-select", placeholder="Stimulus", width="110px"),
                                html.Span("→", style={"color": muted}),
                                _dropdown("op-stim-area", placeholder="Area", width="110px"),
                                _btn("Go", "btn-stimulate", color="#4a3a2a"),
                            ]),

                            # Project
                            html.Div("Project:", style={"color": muted,
                                        "fontSize": "12px", "marginBottom": "4px"}),
                            html.Div(style={"display": "flex", "gap": "6px",
                                            "alignItems": "center", "marginBottom": "10px"},
                                     children=[
                                _dropdown("op-proj-src", placeholder="From", width="100px"),
                                html.Span("→", style={"color": muted}),
                                _dropdown("op-proj-dst", placeholder="To", width="100px"),
                                _btn("Go", "btn-project", color="#3a3a5a"),
                            ]),

                            # Reciprocal
                            html.Div("Reciprocal:", style={"color": muted,
                                        "fontSize": "12px", "marginBottom": "4px"}),
                            html.Div(style={"display": "flex", "gap": "6px",
                                            "alignItems": "center", "marginBottom": "10px"},
                                     children=[
                                _dropdown("op-recip-a", placeholder="Area A", width="100px"),
                                html.Span("↔", style={"color": muted}),
                                _dropdown("op-recip-b", placeholder="Area B", width="100px"),
                                _btn("Go", "btn-reciprocal", color="#3a3a5a"),
                            ]),

                            # Associate
                            html.Div("Associate (2 stimuli → 1 area):",
                                     style={"color": muted,
                                        "fontSize": "12px", "marginBottom": "4px"}),
                            html.Div(style={"display": "flex", "gap": "6px",
                                            "alignItems": "center", "marginBottom": "10px"},
                                     children=[
                                _dropdown("op-assoc-stim-a", placeholder="Stim A", width="90px"),
                                html.Span("+", style={"color": muted}),
                                _dropdown("op-assoc-stim-b", placeholder="Stim B", width="90px"),
                                html.Span("→", style={"color": muted}),
                                _dropdown("op-assoc-area", placeholder="Area", width="80px"),
                                _btn("Go", "btn-associate", color="#4a2a4a"),
                            ]),

                            # Merge
                            html.Div("Merge (2 areas → 1 area):",
                                     style={"color": muted,
                                        "fontSize": "12px", "marginBottom": "4px"}),
                            html.Div(style={"display": "flex", "gap": "6px",
                                            "alignItems": "center", "marginBottom": "10px"},
                                     children=[
                                _dropdown("op-merge-a", placeholder="Area A", width="80px"),
                                html.Span("+", style={"color": muted}),
                                _dropdown("op-merge-b", placeholder="Area B", width="80px"),
                                html.Span("→", style={"color": muted}),
                                _dropdown("op-merge-dst", placeholder="Target", width="80px"),
                                _btn("Go", "btn-merge", color="#2a3a4a"),
                            ]),

                            # Rounds
                            html.Div(style={"display": "flex", "gap": "6px",
                                            "alignItems": "center"},
                                     children=[
                                html.Span("Rounds:", style={"color": muted, "fontSize": "12px"}),
                                _input("op-rounds", "", type="number", value=1, width="50px"),
                                _btn("×10", "btn-x10", color="#4a3a2a"),
                                _btn("×50", "btn-x50", color="#4a2a2a"),
                            ]),
                        ]),

                        # 4. Learning toggle
                        _panel("🔒 Learning Mode", [
                            html.Div(style={"display": "flex", "gap": "10px",
                                            "alignItems": "center"},
                                     children=[
                                _btn("🟢 Learning ON", "btn-learn-on", color="#2a4a2a"),
                                _btn("🔴 Learning OFF (β=0)", "btn-learn-off", color="#4a2a2a"),
                            ]),
                            html.Div(id="learning-status",
                                     style={"fontSize": "12px", "marginTop": "6px",
                                            "color": "#4ade80"}),
                        ]),

                        # 5. Clear / Reset
                        _panel("🗑️ Reset", [
                            _btn("Clear All — Reset Brain", "btn-clear", color="#7f1d1d"),
                            html.Div(id="clear-msg", style={"fontSize": "11px",
                                                            "color": "#f87171", "marginTop": "4px"}),
                        ]),

                        # 6. Save / Load
                        _panel("💾 Save / Load", [
                            dcc.Input(id="save-name", type="text",
                                      placeholder="Save name (e.g. my_brain)",
                                      style={"width": "100%", "marginBottom": "6px",
                                             "backgroundColor": "#1a1a2e", "color": "#ccc",
                                             "border": "1px solid #333", "borderRadius": "4px",
                                             "padding": "4px 8px", "fontSize": "12px"}),
                            html.Div(style={"display": "flex", "gap": "6px"}, children=[
                                _btn("💾 Save", "btn-save", color="#1a4a2a"),
                                _btn("📂 Load", "btn-load", color="#2a3a4a"),
                            ]),
                            html.Div(id="save-msg", style={"fontSize": "11px",
                                                           "color": "#4ade80", "marginTop": "4px"}),
                        ]),

                        # 7. DFA Trainer
                        _panel("🎯 DFA Trainer", [
                            # Setup row
                            html.Div("Setup (resets brain):",
                                     style={"color": muted, "fontSize": "12px",
                                            "marginBottom": "4px"}),
                            html.Div(style={"display": "flex", "gap": "6px",
                                            "marginBottom": "8px"}, children=[
                                dcc.Input(id="dfa-alphabet", type="text",
                                          value="ab", placeholder="Alphabet",
                                          style={"width": "60px",
                                                 "backgroundColor": "#1a1a2e",
                                                 "color": "#ccc",
                                                 "border": "1px solid #333",
                                                 "borderRadius": "4px",
                                                 "padding": "4px 6px",
                                                 "fontSize": "12px"}),
                                _btn("⚙️ Setup DFA", "btn-dfa-setup",
                                     color="#2a3a2a"),
                            ]),

                            # Training strings
                            html.Div("Accepted strings (comma-sep):",
                                     style={"color": muted, "fontSize": "12px",
                                            "marginBottom": "2px"}),
                            dcc.Input(id="dfa-accepted", type="text",
                                      value="a,ab,abb,abbb",
                                      placeholder="a,ab,abb,...",
                                      style={"width": "100%",
                                             "marginBottom": "6px",
                                             "backgroundColor": "#1a1a2e",
                                             "color": "#ccc",
                                             "border": "1px solid #333",
                                             "borderRadius": "4px",
                                             "padding": "4px 8px",
                                             "fontSize": "12px"}),
                            html.Div("Rejected strings (comma-sep):",
                                     style={"color": muted, "fontSize": "12px",
                                            "marginBottom": "2px"}),
                            dcc.Input(id="dfa-rejected", type="text",
                                      value="b,ba,aa,bb,bba",
                                      placeholder="b,ba,aa,...",
                                      style={"width": "100%",
                                             "marginBottom": "6px",
                                             "backgroundColor": "#1a1a2e",
                                             "color": "#ccc",
                                             "border": "1px solid #333",
                                             "borderRadius": "4px",
                                             "padding": "4px 8px",
                                             "fontSize": "12px"}),

                            # Training controls
                            html.Div(style={"display": "flex", "gap": "6px",
                                            "alignItems": "center",
                                            "marginBottom": "8px"}, children=[
                                html.Span("Epochs:", style={"color": muted,
                                                            "fontSize": "12px"}),
                                dcc.Input(id="dfa-epochs", type="number",
                                          value=5, min=1, max=50,
                                          style={"width": "50px",
                                                 "backgroundColor": "#1a1a2e",
                                                 "color": "#ccc",
                                                 "border": "1px solid #333",
                                                 "borderRadius": "4px",
                                                 "padding": "4px",
                                                 "fontSize": "12px"}),
                                _btn("🏋️ Train", "btn-dfa-train",
                                     color="#4a3a1a"),
                            ]),

                            html.Div(id="dfa-train-msg", style={
                                "fontSize": "11px", "color": "#fbbf24",
                                "marginBottom": "8px"}),

                            # Testing
                            html.Div("Test string:",
                                     style={"color": muted, "fontSize": "12px",
                                            "marginBottom": "2px"}),
                            html.Div(style={"display": "flex", "gap": "6px",
                                            "marginBottom": "6px"}, children=[
                                dcc.Input(id="dfa-test-str", type="text",
                                          placeholder="e.g. abbb",
                                          style={"width": "120px",
                                                 "backgroundColor": "#1a1a2e",
                                                 "color": "#ccc",
                                                 "border": "1px solid #333",
                                                 "borderRadius": "4px",
                                                 "padding": "4px 8px",
                                                 "fontSize": "12px"}),
                                _btn("🧪 Test", "btn-dfa-test",
                                     color="#1a3a4a"),
                            ]),
                            html.Div(id="dfa-test-result", style={
                                "fontSize": "12px",
                                "padding": "6px",
                                "borderRadius": "4px",
                                "backgroundColor": "#0f0f1a"}),
                        ]),

                        # 8. Operation log
                        _panel("📝 Log", [
                            html.Div(id="op-log",
                                     style={"maxHeight": "200px", "overflowY": "auto",
                                            "fontSize": "11px", "fontFamily": "monospace"}),
                        ]),
                    ]),

                    # ── RIGHT: Visualization ─────────────────────────
                    html.Div([
                        # Row 1: Graph + Area Table
                        html.Div(
                            style={"display": "grid",
                                   "gridTemplateColumns": "1fr 1fr",
                                   "gap": "12px"},
                            children=[
                                _panel("🕸️ Brain Graph  —  click an area to inspect", [
                                    html.Div(id="graph-container",
                                             style={"minHeight": "320px"}),
                                    html.Div(id="neuron-detail",
                                             style={"marginTop": "8px"}),
                                ]),
                                _panel("📋 Area Status", [
                                    html.Div(id="area-table",
                                             style={"maxHeight": "340px",
                                                    "overflowY": "auto"}),
                                ]),
                            ],
                        ),

                        # Row 2: Convergence Chart + Weight Histogram
                        html.Div(
                            style={"display": "grid",
                                   "gridTemplateColumns": "1fr 1fr",
                                   "gap": "12px"},
                            children=[
                                _panel("📈 Convergence (Overlap with Previous)", [
                                    _dropdown("conv-area", placeholder="Area",
                                              width="140px"),
                                    dcc.Graph(id="convergence-chart",
                                              config={"displayModeBar": False},
                                              style={"height": "240px"}),
                                ]),
                                _panel("📊 Weight Distribution", [
                                    html.Div(
                                        style={"display": "flex", "gap": "6px",
                                               "marginBottom": "6px"},
                                        children=[
                                            _dropdown("wt-from", placeholder="From",
                                                      width="110px"),
                                            html.Span("→", style={"color": muted}),
                                            _dropdown("wt-to", placeholder="To",
                                                      width="110px"),
                                        ],
                                    ),
                                    dcc.Graph(id="weight-histogram",
                                              config={"displayModeBar": False},
                                              style={"height": "220px"}),
                                ]),
                            ],
                        ),
                    ]),
                ],
            ),

            # Hidden stores for triggering updates
            dcc.Store(id="refresh-trigger", data=0),
            dcc.Store(id="selected-area", data=None),
        ],
    )

    # ── Callbacks ────────────────────────────────────────────────────────

    # == Add Area ==
    @app.callback(
        Output("area-msg", "children"),
        Output("refresh-trigger", "data", allow_duplicate=True),
        Input("btn-add-area", "n_clicks"),
        State("area-name", "value"),
        State("area-n", "value"),
        State("area-k", "value"),
        State("area-beta", "value"),
        State("refresh-trigger", "data"),
        prevent_initial_call=True,
    )
    def add_area(n_clicks, name, n, k, beta, refresh):
        if not name or not name.strip():
            return "⚠ Enter area name.", refresh
        name = name.strip()
        if name in _brain.get_areas():
            return f"⚠ Area '{name}' exists.", refresh
        n = int(n or 500)
        k = int(k or 50)
        beta = float(beta or 0.05)
        _brain.add_area(name, n=n, k=k, beta=beta)
        return f"✅ Added '{name}' (n={n}, k={k})", refresh + 1

    # == Add Stimulus ==
    @app.callback(
        Output("stim-msg", "children"),
        Output("refresh-trigger", "data", allow_duplicate=True),
        Input("btn-add-stim", "n_clicks"),
        State("stim-name", "value"),
        State("stim-size", "value"),
        State("refresh-trigger", "data"),
        prevent_initial_call=True,
    )
    def add_stimulus(n_clicks, name, size, refresh):
        if not name or not name.strip():
            return "⚠ Enter stimulus name.", refresh
        name = name.strip()
        size = int(size or 100)
        _brain.add_stimulus(name, size=size)
        return f"✅ Added stimulus '{name}' (size={size})", refresh + 1

    # == Stimulate ==
    @app.callback(
        Output("refresh-trigger", "data", allow_duplicate=True),
        Input("btn-stimulate", "n_clicks"),
        State("op-stim-select", "value"),
        State("op-stim-area", "value"),
        State("op-rounds", "value"),
        State("refresh-trigger", "data"),
        prevent_initial_call=True,
    )
    def stimulate(n_clicks, stim, area, rounds, refresh):
        if not stim or not area:
            return refresh
        rounds = int(rounds or 1)
        _brain.stimulate(stim, area, rounds=rounds)
        return refresh + 1

    # == Project ==
    @app.callback(
        Output("refresh-trigger", "data", allow_duplicate=True),
        Input("btn-project", "n_clicks"),
        State("op-proj-src", "value"),
        State("op-proj-dst", "value"),
        State("op-rounds", "value"),
        State("refresh-trigger", "data"),
        prevent_initial_call=True,
    )
    def project(n_clicks, src, dst, rounds, refresh):
        if not src or not dst:
            return refresh
        rounds = int(rounds or 1)
        _brain.project(src, dst, rounds=rounds)
        return refresh + 1

    # == Reciprocal ==
    @app.callback(
        Output("refresh-trigger", "data", allow_duplicate=True),
        Input("btn-reciprocal", "n_clicks"),
        State("op-recip-a", "value"),
        State("op-recip-b", "value"),
        State("op-rounds", "value"),
        State("refresh-trigger", "data"),
        prevent_initial_call=True,
    )
    def reciprocal(n_clicks, a, b, rounds, refresh):
        if not a or not b:
            return refresh
        rounds = int(rounds or 1)
        _brain.reciprocal_project(a, b, rounds=rounds)
        return refresh + 1

    # == Associate ==
    @app.callback(
        Output("refresh-trigger", "data", allow_duplicate=True),
        Input("btn-associate", "n_clicks"),
        State("op-assoc-stim-a", "value"),
        State("op-assoc-stim-b", "value"),
        State("op-assoc-area", "value"),
        State("op-rounds", "value"),
        State("refresh-trigger", "data"),
        prevent_initial_call=True,
    )
    def associate(n_clicks, stim_a, stim_b, area, rounds, refresh):
        if not stim_a or not stim_b or not area:
            return refresh
        rounds = int(rounds or 1)
        _brain.associate(stim_a, stim_b, area, rounds=rounds)
        return refresh + 1

    # == Merge ==
    @app.callback(
        Output("refresh-trigger", "data", allow_duplicate=True),
        Input("btn-merge", "n_clicks"),
        State("op-merge-a", "value"),
        State("op-merge-b", "value"),
        State("op-merge-dst", "value"),
        State("op-rounds", "value"),
        State("refresh-trigger", "data"),
        prevent_initial_call=True,
    )
    def merge_op(n_clicks, src_a, src_b, dst, rounds, refresh):
        if not src_a or not src_b or not dst:
            return refresh
        rounds = int(rounds or 1)
        _brain.merge(src_a, src_b, dst, rounds=rounds)
        return refresh + 1

    # == ×10 / ×50 quick buttons ==
    @app.callback(
        Output("op-rounds", "value"),
        Input("btn-x10", "n_clicks"),
        Input("btn-x50", "n_clicks"),
        prevent_initial_call=True,
    )
    def set_rounds(x10, x50):
        triggered = dash.ctx.triggered_id
        if triggered == "btn-x10":
            return 10
        elif triggered == "btn-x50":
            return 50
        return 1

    # == Learning toggle ==
    @app.callback(
        Output("learning-status", "children"),
        Input("btn-learn-on", "n_clicks"),
        Input("btn-learn-off", "n_clicks"),
        prevent_initial_call=True,
    )
    def toggle_learning(on_clicks, off_clicks):
        triggered = dash.ctx.triggered_id
        if triggered == "btn-learn-on":
            _brain.set_learning(True)
            return "🟢 Learning ENABLED — weights will update"
        elif triggered == "btn-learn-off":
            _brain.set_learning(False)
            return "🔴 Learning DISABLED — β=0, weights frozen"
        return ""

    # == Clear / Reset ==
    @app.callback(
        Output("clear-msg", "children"),
        Output("refresh-trigger", "data", allow_duplicate=True),
        Input("btn-clear", "n_clicks"),
        State("refresh-trigger", "data"),
        prevent_initial_call=True,
    )
    def clear_brain(n_clicks, refresh):
        _brain.reset()
        return "🗑️ Brain reset.", refresh + 1

    # == Save brain state ==
    @app.callback(
        Output("save-msg", "children"),
        Input("btn-save", "n_clicks"),
        State("save-name", "value"),
        prevent_initial_call=True,
    )
    def save_brain(n_clicks, name):
        if not name or not name.strip():
            return "⚠️ Enter a save name first"
        name = name.strip().replace(" ", "_")
        save_dir = os.path.join(os.path.dirname(__file__), "..", "saves", name)
        try:
            _brain.save_state(save_dir)
            return f"💾 Saved to saves/{name}/"
        except Exception as e:
            return f"❌ Save failed: {e}"

    # == Load brain state ==
    @app.callback(
        Output("save-msg", "children", allow_duplicate=True),
        Output("refresh-trigger", "data", allow_duplicate=True),
        Input("btn-load", "n_clicks"),
        State("save-name", "value"),
        State("refresh-trigger", "data"),
        prevent_initial_call=True,
    )
    def load_brain(n_clicks, name, refresh):
        if not name or not name.strip():
            return "⚠️ Enter a save name to load", refresh
        name = name.strip().replace(" ", "_")
        save_dir = os.path.join(os.path.dirname(__file__), "..", "saves", name)
        try:
            _brain.load_state(save_dir)
            return f"📂 Loaded from saves/{name}/", refresh + 1
        except FileNotFoundError:
            return f"❌ Save '{name}' not found", refresh
        except Exception as e:
            return f"❌ Load failed: {e}", refresh

    # == DFA Setup ==
    @app.callback(
        Output("dfa-train-msg", "children"),
        Output("refresh-trigger", "data", allow_duplicate=True),
        Input("btn-dfa-setup", "n_clicks"),
        State("dfa-alphabet", "value"),
        State("refresh-trigger", "data"),
        prevent_initial_call=True,
    )
    def dfa_setup(n_clicks, alphabet, refresh):
        if not alphabet or not alphabet.strip():
            return "⚠️ Enter alphabet characters", refresh
        alphabet = alphabet.strip()
        try:
            _brain.setup_dfa(alphabet=alphabet)
            return (f"⚙️ DFA ready: alphabet='{alphabet}', "
                    f"areas: Input, State, Accept, Reject"), refresh + 1
        except Exception as e:
            return f"❌ Setup failed: {e}", refresh

    # == DFA Train ==
    @app.callback(
        Output("dfa-train-msg", "children", allow_duplicate=True),
        Output("refresh-trigger", "data", allow_duplicate=True),
        Input("btn-dfa-train", "n_clicks"),
        State("dfa-accepted", "value"),
        State("dfa-rejected", "value"),
        State("dfa-epochs", "value"),
        State("refresh-trigger", "data"),
        prevent_initial_call=True,
    )
    def dfa_train(n_clicks, accepted_str, rejected_str, epochs, refresh):
        if not accepted_str and not rejected_str:
            return "⚠️ Enter some training strings", refresh
        accepted = [s.strip() for s in (accepted_str or "").split(",")
                    if s.strip()]
        rejected = [s.strip() for s in (rejected_str or "").split(",")
                    if s.strip()]
        epochs = max(1, int(epochs or 5))
        try:
            result = _brain.train_batch(accepted, rejected, epochs=epochs)
            return (f"🏋️ Trained! {result['epochs']} epochs × "
                    f"{result['accepted_count']+result['rejected_count']} strings = "
                    f"{result['total_trainings']} rounds"), refresh + 1
        except Exception as e:
            return f"❌ Train failed: {e}", refresh

    # == DFA Test ==
    @app.callback(
        Output("dfa-test-result", "children"),
        Output("refresh-trigger", "data", allow_duplicate=True),
        Input("btn-dfa-test", "n_clicks"),
        State("dfa-test-str", "value"),
        State("refresh-trigger", "data"),
        prevent_initial_call=True,
    )
    def dfa_test(n_clicks, test_str, refresh):
        if not test_str or not test_str.strip():
            return "⚠️ Enter a test string", refresh
        test_str = test_str.strip()
        try:
            result = _brain.test_string(test_str)
            v = result["verdict"]
            if v == "ACCEPT":
                color, emoji = "#4ade80", "✅"
            elif v == "REJECT":
                color, emoji = "#f87171", "❌"
            else:
                color, emoji = "#fbbf24", "❓"
            return html.Div([
                html.Span(f"{emoji} '{test_str}' → {v}",
                          style={"color": color, "fontWeight": "bold",
                                 "fontSize": "13px"}),
                html.Br(),
                html.Span(
                    f"Accept overlap: {result['accept_overlap']:.4f} | "
                    f"Reject overlap: {result['reject_overlap']:.4f} | "
                    f"Confidence: {result['confidence']:.2%}",
                    style={"color": "#888", "fontSize": "11px"}),
            ]), refresh + 1
        except Exception as e:
            return f"❌ Test failed: {e}", refresh

    # == Master refresh: update all dropdowns + status + viz ==
    @app.callback(
        Output("status-bar", "children"),
        Output("op-stim-select", "options"),
        Output("op-stim-area", "options"),
        Output("op-proj-src", "options"),
        Output("op-proj-dst", "options"),
        Output("op-recip-a", "options"),
        Output("op-recip-b", "options"),
        Output("op-assoc-stim-a", "options"),
        Output("op-assoc-stim-b", "options"),
        Output("op-assoc-area", "options"),
        Output("op-merge-a", "options"),
        Output("op-merge-b", "options"),
        Output("op-merge-dst", "options"),
        Output("conv-area", "options"),
        Output("wt-from", "options"),
        Output("wt-to", "options"),
        Output("graph-container", "children"),
        Output("area-table", "children"),
        Output("op-log", "children"),
        Input("refresh-trigger", "data"),
    )
    def refresh_all(trigger):
        areas = _brain.get_areas()
        stimuli = _brain.get_stimuli()

        area_opts = [{"label": a, "value": a} for a in areas]
        stim_opts = [{"label": s, "value": s} for s in stimuli]

        # Status bar
        n_steps = len(_brain.steps)
        # Use last snapshot for consensus on learning state and stats
        last_snap = _brain._last_snapshot
        learn_str = "ON" if _brain.learning else "OFF (β=0)"
        status = f"Areas: {len(areas)} | Stimuli: {len(stimuli)} | Steps: {n_steps} | Learning: {learn_str}"

        # Graph
        graph = _render_graph(areas, last_snap)

        # Area table
        table = _render_area_table()

        # Operation log
        log = _render_log()

        # Weight chart "From" includes both stimuli and areas
        wt_from_opts = (
            [{"label": f"⚡ {s}", "value": s} for s in stimuli] +
            [{"label": a, "value": a} for a in areas]
        )

        return (status, stim_opts, area_opts, area_opts, area_opts,
                area_opts, area_opts,
                stim_opts, stim_opts, area_opts,  # associate dropdowns
                area_opts, area_opts, area_opts,   # merge dropdowns
                area_opts, wt_from_opts, area_opts,
                graph, table, log)

    # == Convergence chart ==
    @app.callback(
        Output("convergence-chart", "figure"),
        Input("conv-area", "value"),
        Input("refresh-trigger", "data"),
    )
    def update_convergence(area, trigger):
        return _render_convergence(area)

    # == Weight histogram ==
    @app.callback(
        Output("weight-histogram", "figure"),
        Input("wt-from", "value"),
        Input("wt-to", "value"),
        Input("refresh-trigger", "data"),
    )
    def update_weights(from_area, to_area, trigger):
        return _render_weight_histogram(from_area, to_area)

    # == Cytoscape node click → store selected area ==
    @app.callback(
        Output("selected-area", "data"),
        Input("cyto-graph", "tapNodeData"),
        prevent_initial_call=True,
    )
    def on_node_click(node_data):
        if node_data and node_data.get("id"):
            return node_data["id"]
        # Don't clear selection when graph re-renders
        raise dash.exceptions.PreventUpdate

    # == Render neuron detail for selected area ==
    @app.callback(
        Output("neuron-detail", "children"),
        Input("selected-area", "data"),
        Input("refresh-trigger", "data"),
    )
    def update_neuron_detail(area_name, trigger):
        return _render_neuron_detail(area_name)

    return app


# ── Render Helpers ───────────────────────────────────────────────────────────

_CHART_LAYOUT = dict(
    plot_bgcolor="#1a1a2e",
    paper_bgcolor="#1a1a2e",
    font={"color": "#aaa", "size": 10},
    margin={"l": 35, "r": 15, "t": 15, "b": 30},
)


def _render_neuron_detail(area_name):
    """Render a visual neuron grid for the selected area."""
    if not area_name:
        return html.Div("Click an area node above to inspect neurons.",
                        style={"color": "#666", "fontSize": "12px",
                               "padding": "8px", "textAlign": "center"})

    area = _brain.brain.area_by_name.get(area_name)
    if not area:
        return html.Div(f"Area '{area_name}' not found.",
                        style={"color": "#f87171", "fontSize": "12px"})

    state = _brain._last_snapshot.get(area_name, _brain.get_area_state(area_name))
    winners = state.get("winners_set", set())
    gained = state.get("gained_set", set())
    lost = state.get("lost_set", set())
    n = area.n

    # Assembly status
    overlap = state.get("overlap_prev", 0)
    if not winners:
        asm_label = "⚪ No assembly"
        asm_color = "#666"
    elif overlap >= 0.9:
        asm_label = f"🟢 Assembly CONVERGED ({overlap:.0%} stable)"
        asm_color = "#4ade80"
    elif overlap >= 0.5:
        asm_label = f"🟡 Assembly FORMING ({overlap:.0%})"
        asm_color = "#fbbf24"
    else:
        asm_label = f"🟠 Assembly UNSTABLE ({overlap:.0%})"
        asm_color = "#fb923c"

    # Summary header
    header = html.Div([
        html.Span(f"Area '{area_name}'", style={
            "fontWeight": "700", "fontSize": "14px", "marginRight": "12px"}),
        html.Span(asm_label, style={"color": asm_color, "fontSize": "12px",
                                     "marginRight": "12px"}),
        html.Span(f"🟢 {len(winners)} winners", style={
            "color": "#4ade80", "fontSize": "11px", "marginRight": "8px"}),
        html.Span(f"🔺 +{len(gained)} gained", style={
            "color": "#22d3ee", "fontSize": "11px", "marginRight": "8px"}),
        html.Span(f"🔻 -{len(lost)} lost", style={
            "color": "#f87171", "fontSize": "11px", "marginRight": "8px"}),
        html.Span(f"⚫ {n - len(winners)} inactive", style={
            "color": "#666", "fontSize": "11px"}),
    ], style={"marginBottom": "8px"})

    # Neuron grid — show up to 500 neurons
    max_display = min(n, 500)
    dots = []
    for i in range(max_display):
        if i in gained:
            # Newly gained — bright cyan
            bg = "#22d3ee"
            border = "1px solid #06b6d4"
            title = f"Neuron {i} — GAINED (new winner)"
        elif i in winners:
            # Stable winner — green
            bg = "#4ade80"
            border = "1px solid #22c55e"
            title = f"Neuron {i} — WINNER"
        elif i in lost:
            # Lost — was winner, no longer
            bg = "#f87171"
            border = "1px solid #ef4444"
            title = f"Neuron {i} — LOST (was winner)"
        else:
            # Not firing
            bg = "#2a2a3a"
            border = "1px solid #1a1a2a"
            title = f"Neuron {i} — inactive"

        dots.append(html.Div(
            title=title,
            style={
                "width": "6px", "height": "6px",
                "backgroundColor": bg,
                "border": border,
                "borderRadius": "1px",
                "display": "inline-block",
            },
        ))

    suffix = ""
    if n > max_display:
        suffix = f"  … showing {max_display}/{n} neurons"

    grid = html.Div(
        dots,
        style={"display": "flex", "flexWrap": "wrap", "gap": "2px"},
    )

    footer = html.Div(suffix, style={
        "color": "#666", "fontSize": "10px", "marginTop": "4px",
    }) if suffix else None

    children = [header, grid]
    if footer:
        children.append(footer)

    return html.Div(children, style={
        "backgroundColor": "#0f0f1a", "borderRadius": "8px",
        "padding": "10px", "border": "1px solid #2a2a4a",
    })


def _render_graph(areas, state_override=None):
    """Render live Cytoscape graph of brain areas."""
    if not areas:
        return html.Div("Add areas to see the graph.",
                        style={"color": "#666", "fontSize": "13px",
                               "padding": "40px", "textAlign": "center"})

    # Use override (e.g. last_snapshot) or fall back to full state
    state = state_override or _brain.get_full_state()
    fibers = _brain.get_fibers()

    elements = []
    n_areas = len(areas)
    for i, area_name in enumerate(areas):
        # Arrange in circle
        angle = 2 * math.pi * i / max(n_areas, 1)
        x = 200 + 120 * math.cos(angle)
        y = 200 + 120 * math.sin(angle)

        area_state = state.get(area_name, {})
        n_winners = area_state.get("n_winners", 0)
        overlap = area_state.get("overlap_prev", 0)

        # Node styling based on activation
        classes = ["area"]
        if n_winners > 0:
            classes.append("active")
            if overlap > 0.9:
                classes.append("converged")

        label = f"{area_name}\n{n_winners}w"
        if n_winners > 0:
            label += f"\n{overlap:.0%}"

        elements.append({
            "data": {"id": area_name, "label": label},
            "position": {"x": x, "y": y},
            "classes": " ".join(classes),
        })

    # Add fiber edges
    for src, dst in fibers:
        if src in areas and dst in areas:
            src_state = state.get(src, {})
            classes = "active" if src_state.get("n_winners", 0) > 0 else ""
            elements.append({
                "data": {"source": src, "target": dst, "label": ""},
                "classes": classes,
            })

    stylesheet = [
        {"selector": "node",
         "style": {"label": "data(label)",
                   "background-color": "#2a2a4a",
                   "color": "#ddd", "font-size": "10px",
                   "text-valign": "center", "text-halign": "center",
                   "width": "60px", "height": "60px",
                   "border-width": "2px", "border-color": "#4a4a6a",
                   "text-wrap": "wrap", "text-max-width": "55px"}},
        {"selector": ".active",
         "style": {"background-color": "#2a4a2a",
                   "border-color": "#4ade80",
                   "border-width": "3px"}},
        {"selector": ".converged",
         "style": {"background-color": "#1a3a1a",
                   "border-color": "#22c55e",
                   "border-width": "4px"}},
        {"selector": "edge",
         "style": {"width": 1, "line-color": "#3a3a5a",
                   "curve-style": "bezier",
                   "target-arrow-shape": "triangle",
                   "target-arrow-color": "#3a3a5a",
                   "arrow-scale": 0.6}},
        {"selector": "edge.active",
         "style": {"line-color": "#6366f1", "width": 2,
                   "target-arrow-color": "#6366f1"}},
    ]

    import dash_cytoscape as cyto
    return cyto.Cytoscape(
        id="cyto-graph",
        elements=elements,
        stylesheet=stylesheet,
        layout={"name": "preset"},
        style={"width": "100%", "height": "320px",
               "backgroundColor": "#0f0f1a"},
    )


def _render_area_table():
    """Render area status table."""
    areas = _brain.get_areas()
    if not areas:
        return html.Div("No areas yet.", style={"color": "#666"})

    rows = []
    # Header
    rows.append(html.Tr([
        html.Th(h, style={"padding": "4px 8px", "borderBottom": "1px solid #3a3a5a",
                          "color": "#aaa", "fontSize": "11px", "textAlign": "left"})
        for h in ["Area", "n", "k", "Winners", "Overlap", "Gained", "Lost", "β"]
    ]))

    for name in areas:
        state = _brain._last_snapshot.get(name, _brain.get_area_state(name))
        overlap = state.get("overlap_prev", 0)
        n_w = state.get("n_winners", 0)

        # Color code overlap
        if n_w == 0:
            ov_color = "#666"
        elif overlap >= 0.9:
            ov_color = "#4ade80"
        elif overlap >= 0.5:
            ov_color = "#fbbf24"
        else:
            ov_color = "#f87171"

        row_style = {"padding": "3px 8px", "fontSize": "12px",
                     "borderBottom": "1px solid #1e1e3a", "color": "#ddd"}

        rows.append(html.Tr([
            html.Td(name, style={**row_style, "fontWeight": "600"}),
            html.Td(str(state.get("n", 0)), style=row_style),
            html.Td(str(state.get("k", 0)), style=row_style),
            html.Td(str(n_w), style={**row_style, "color": "#4ade80" if n_w > 0 else "#666"}),
            html.Td(f"{overlap:.0%}", style={**row_style, "color": ov_color}),
            html.Td(f"+{state.get('gained', 0)}", style={**row_style, "color": "#4ade80"}),
            html.Td(f"-{state.get('lost', 0)}", style={**row_style, "color": "#f87171"}),
            html.Td(f"{state.get('beta', 0):.2f}", style={**row_style, "color": "#aaa"}),
        ]))

    return html.Table(rows, style={"width": "100%", "borderCollapse": "collapse"})


def _render_log():
    """Render operation log (last 30 steps)."""
    steps = _brain.steps[-30:]
    if not steps:
        return html.Div("No operations yet.", style={"color": "#666"})

    items = []
    for step in reversed(steps):
        learn_icon = "🟢" if step.learning else "🔴"
        items.append(html.Div(
            f"{learn_icon} [{step.step}] {step.description}",
            style={"color": "#ccc", "padding": "2px 0",
                   "borderBottom": "1px solid #1a1a2e"},
        ))
    return html.Div(items)


def _render_convergence(area):
    """Render convergence chart for an area."""
    fig = go.Figure()

    if area:
        history = _brain.get_overlap_history(area)
        if history:
            fig.add_trace(go.Scatter(
                y=history,
                mode="lines+markers",
                name="Overlap with prev",
                line={"color": "#10b981", "width": 2},
                marker={"size": 4},
                fill="tozeroy",
                fillcolor="rgba(16, 185, 129, 0.1)",
            ))

            # Convergence threshold
            fig.add_hline(y=0.9, line_dash="dot", line_color="#fbbf24",
                         opacity=0.7,
                         annotation_text="90% threshold",
                         annotation_font_size=9,
                         annotation_font_color="#fbbf24")

        winner_history = _brain.get_winner_count_history(area)
        if winner_history:
            max_w = max(winner_history) or 1
            normalized = [w / max_w for w in winner_history]
            fig.add_trace(go.Scatter(
                y=normalized,
                mode="lines",
                name="Winners (normalized)",
                line={"color": "#6366f1", "width": 1, "dash": "dot"},
            ))

    fig.update_layout(
        **_CHART_LAYOUT,
        xaxis={"gridcolor": "#2a2a4a", "title": "Step", "title_font_size": 10},
        yaxis={"gridcolor": "#2a2a4a", "range": [-0.05, 1.1],
               "title": "Value", "title_font_size": 10},
        legend={"orientation": "h", "y": 1.15, "font": {"size": 9}},
        height=220,
    )
    return fig


def _render_weight_histogram(from_area, to_area):
    """Render weight distribution chart — split into unlearned vs strengthened."""
    fig = go.Figure()
    title_text = "Select areas above"

    if from_area and to_area:
        sample = _brain.get_weight_sample(from_area, to_area, sample_size=5000)
        if sample:
            nonzero = [w for w in sample if w > 0]

            if nonzero:
                stats = _brain.get_weight_stats(from_area, to_area)
                mean_w = stats.get("mean_nonzero", 1.0)
                max_w = stats.get("max", 1.0)
                min_w = min(nonzero)

                # Split into unlearned (≈1.0) and strengthened (>1.0)
                unlearned = [w for w in nonzero if w <= 1.001]
                strengthened = [w for w in nonzero if w > 1.001]

                if unlearned:
                    fig.add_trace(go.Histogram(
                        x=unlearned, nbinsx=20,
                        name=f"Unlearned ({len(unlearned)})",
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

                # Summary line
                pct_strong = len(strengthened) / len(nonzero) * 100
                growth = (max_w - 1.0) / 1.0 * 100  # % growth from initial
                title_text = (
                    f"{from_area} → {to_area}  |  "
                    f"{len(strengthened)}/{len(nonzero)} strengthened ({pct_strong:.0f}%)  |  "
                    f"max weight: {max_w:.3f} (+{growth:.0f}% from initial)"
                )
            else:
                title_text = f"{from_area} → {to_area}  |  No connections"
        else:
            title_text = f"{from_area} → {to_area}  |  No weight data"

    fig.update_layout(
        **_CHART_LAYOUT,
        xaxis={"gridcolor": "#2a2a4a", "title": "Synapse Weight",
               "title_font_size": 10},
        yaxis={"gridcolor": "#2a2a4a", "title": "Synapses", "title_font_size": 10},
        height=240,
        barmode="overlay",
        legend={"font": {"size": 9}, "x": 0.7, "y": 0.95,
                "bgcolor": "rgba(0,0,0,0.5)"},
        title={"text": title_text, "font": {"size": 10, "color": "#aaa"},
               "x": 0.5, "xanchor": "center"},
    )
    return fig


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = create_app()
    print("🧠 Assembly Calculus Workbench starting...")
    print("   Open http://127.0.0.1:8050 in your browser")
    app.run(debug=True, port=8050)
