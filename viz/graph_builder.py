"""
Cytoscape Graph Builder — Creates interactive multi-level graph views.

Level 0: Area graph (areas as nodes, projections as edges)
Level 1: Assembly view (assemblies within a selected area)
Level 2: Neuron scatter (winner neurons within selected assembly)
"""

import dash_cytoscape as cyto
from dash import html


# ── Cytoscape stylesheet ────────────────────────────────────────────────────

GRAPH_STYLESHEET = [
    # Default node style
    {
        "selector": "node",
        "style": {
            "label": "data(label)",
            "text-valign": "center",
            "text-halign": "center",
            "background-color": "#2a2a4a",
            "color": "#e0e0ee",
            "font-size": "11px",
            "border-width": 2,
            "border-color": "#3a3a5a",
            "width": 60,
            "height": 60,
        },
    },
    # Active node (has winners)
    {
        "selector": "node.active",
        "style": {
            "background-color": "#6366f1",
            "border-color": "#a78bfa",
            "border-width": 3,
            "font-weight": "bold",
        },
    },
    # Current DFA state
    {
        "selector": "node.current-state",
        "style": {
            "background-color": "#4f46e5",
            "border-color": "#c4b5fd",
            "border-width": 4,
            "width": 75,
            "height": 75,
            "font-size": "13px",
            "font-weight": "bold",
        },
    },
    # Dormant node
    {
        "selector": "node.dormant",
        "style": {
            "background-color": "#1a1a2e",
            "border-color": "#2a2a4a",
            "opacity": 0.6,
        },
    },
    # Error state
    {
        "selector": "node.error",
        "style": {
            "background-color": "#7f1d1d",
            "border-color": "#ef4444",
        },
    },
    # Assembly node (Level 1)
    {
        "selector": "node.assembly",
        "style": {
            "shape": "roundrectangle",
            "width": 80,
            "height": 40,
            "font-size": "10px",
        },
    },
    # Neuron node (Level 2)
    {
        "selector": "node.neuron",
        "style": {
            "width": 12,
            "height": 12,
            "font-size": "8px",
            "label": "",
        },
    },
    {
        "selector": "node.neuron-winner",
        "style": {
            "background-color": "#4ade80",
            "border-color": "#22c55e",
            "width": 14,
            "height": 14,
        },
    },
    {
        "selector": "node.neuron-gained",
        "style": {
            "background-color": "#34d399",
            "border-color": "#10b981",
            "border-style": "dashed",
        },
    },
    {
        "selector": "node.neuron-lost",
        "style": {
            "background-color": "#f87171",
            "border-color": "#ef4444",
            "border-style": "dashed",
            "opacity": 0.7,
        },
    },
    # Default edge style
    {
        "selector": "edge",
        "style": {
            "width": 2,
            "line-color": "#3a3a5a",
            "target-arrow-color": "#3a3a5a",
            "target-arrow-shape": "triangle",
            "curve-style": "bezier",
            "opacity": 0.6,
        },
    },
    # Active edge (data flowing)
    {
        "selector": "edge.active",
        "style": {
            "line-color": "#6366f1",
            "target-arrow-color": "#6366f1",
            "width": 3,
            "opacity": 1,
        },
    },
]


# ── Level 0: Area Graph ─────────────────────────────────────────────────────

def build_area_graph(step: dict, area_names: list[str] = None) -> list[dict]:
    """Build Cytoscape elements for Level 0: areas as nodes.

    Args:
        step: A single step dict from the simulation log.
        area_names: Override area names. If None, uses activations keys.

    Returns:
        List of Cytoscape node/edge elements.
    """
    activations = step.get("activations", {})
    current_state = step.get("dfa_state")
    areas = area_names or sorted(activations.keys())

    elements = []

    # Nodes
    node_positions = _calculate_positions(areas)
    for area in areas:
        act = activations.get(area, {})
        n_winners = len(act.get("winners", []))
        overlap = act.get("template_overlap", 0)
        template = act.get("template_name", "")

        # Determine CSS classes
        classes = []
        if area == current_state:
            classes.append("current-state")
        elif n_winners > 0:
            classes.append("active")
        else:
            classes.append("dormant")

        if area == "q_err":
            classes.append("error")

        # Label with stats
        label = area
        if n_winners > 0:
            label += f"\n{n_winners}w"
            if template:
                label += f"\n{overlap:.0%}"

        pos = node_positions.get(area, {"x": 0, "y": 0})
        elements.append({
            "data": {
                "id": area,
                "label": label,
                "n_winners": n_winners,
                "overlap": overlap,
                "template": template,
            },
            "position": pos,
            "classes": " ".join(classes),
        })

    # Edges — derive from DFA structure
    dfa_transitions = _infer_edges(step, areas)
    for src, tgt, label, is_active in dfa_transitions:
        classes = "active" if is_active else ""
        elements.append({
            "data": {
                "source": src,
                "target": tgt,
                "label": label,
            },
            "classes": classes,
        })

    return elements


def _calculate_positions(areas: list[str]) -> dict[str, dict]:
    """Calculate node positions for area graph layout."""
    import math
    positions = {}
    # Place DFA states in a line, Transitions below
    state_areas = [a for a in areas if a != "Transitions"]
    n = len(state_areas)

    for i, area in enumerate(state_areas):
        x = 100 + i * 150
        y = 100
        positions[area] = {"x": x, "y": y}

    if "Transitions" in areas:
        center_x = 100 + (n - 1) * 75
        positions["Transitions"] = {"x": center_x, "y": 250}

    return positions


def _infer_edges(step: dict, areas: list[str]) -> list[tuple]:
    """Infer edges from DFA transition info.
    Returns list of (source, target, label, is_active) tuples."""
    edges = []
    transition = step.get("dfa_transition", "")
    current_state = step.get("dfa_state")

    # Define known DFA edges for ab* DFA
    known_edges = [
        ("q0", "Transitions", "input", False),
        ("Transitions", "q0", "output", False),
        ("Transitions", "q1", "output", False),
        ("Transitions", "q_err", "output", False),
        ("q1", "Transitions", "input", False),
    ]

    # Activate relevant edges based on current step
    for src, tgt, label, _ in known_edges:
        if src in areas and tgt in areas:
            is_active = False
            if transition:
                # If transition involves src → Transitions or Transitions → tgt
                if "Transitions" in (src, tgt):
                    is_active = True
            edges.append((src, tgt, label, is_active))

    return edges


# ── Level 1: Assembly View ───────────────────────────────────────────────────

def build_assembly_view(
    area: str,
    step: dict,
    assembly_tracker=None,
    step_idx: int = 0,
) -> list[dict]:
    """Build Cytoscape elements for Level 1: assemblies within an area.

    Shows template assemblies as distinct nodes with overlap/status info.
    """
    act = step.get("activations", {}).get(area, {})
    if not act:
        return []

    elements = []
    winners = set(act.get("winners", []))
    template = act.get("template_name", "")
    overlap = act.get("template_overlap", 0)
    gained = set(act.get("winners_gained", []))
    lost = set(act.get("winners_lost", []))

    # Main assembly node
    classes = ["assembly"]
    if overlap >= 0.9:
        classes.append("active")
    elif overlap >= 0.5:
        classes.append("active")  # still active but drifting

    label = template or area
    label += f"\n{overlap:.0%} match"
    label += f"\n{len(winners)}w"

    elements.append({
        "data": {
            "id": f"{area}_main",
            "label": label,
            "overlap": overlap,
        },
        "position": {"x": 200, "y": 100},
        "classes": " ".join(classes),
    })

    # Show winner change as sub-nodes if there are gains/losses
    if gained:
        elements.append({
            "data": {
                "id": f"{area}_gained",
                "label": f"+{len(gained)} gained",
            },
            "position": {"x": 100, "y": 200},
            "classes": "assembly",
            "style": {"background-color": "#166534"},
        })
        elements.append({
            "data": {"source": f"{area}_gained", "target": f"{area}_main"},
            "classes": "active",
        })

    if lost:
        elements.append({
            "data": {
                "id": f"{area}_lost",
                "label": f"-{len(lost)} lost",
            },
            "position": {"x": 300, "y": 200},
            "classes": "assembly",
            "style": {"background-color": "#7f1d1d"},
        })
        elements.append({
            "data": {"source": f"{area}_main", "target": f"{area}_lost"},
        })

    # Attribution nodes
    attr = act.get("attribution", {})
    if attr:
        total = sum(v for v in attr.values() if v > 0)
        y_offset = 0
        for src_name, val in sorted(attr.items(), key=lambda x: -x[1]):
            share = val / total if total > 0 else 0
            elements.append({
                "data": {
                    "id": f"{area}_src_{src_name}",
                    "label": f"{src_name}\n{share:.0%}",
                },
                "position": {"x": 400, "y": 80 + y_offset},
                "classes": "assembly active" if share > 0.3 else "assembly dormant",
            })
            elements.append({
                "data": {
                    "source": f"{area}_src_{src_name}",
                    "target": f"{area}_main",
                },
                "classes": "active" if share > 0.3 else "",
            })
            y_offset += 60

    return elements


# ── Level 2: Neuron View ─────────────────────────────────────────────────────

def build_neuron_view(
    area: str,
    step: dict,
    max_neurons: int = 200,
) -> list[dict]:
    """Build Cytoscape elements for Level 2: individual winner neurons.

    Shows winners as green dots, gained as dashed green, lost as dashed red.
    Limits display to max_neurons for performance.
    """
    act = step.get("activations", {}).get(area, {})
    if not act:
        return []

    elements = []
    winners = act.get("winners", [])
    gained = set(act.get("winners_gained", []))
    lost = set(act.get("winners_lost", []))

    # Combine all relevant neurons
    all_neurons = set(winners) | gained | lost
    if len(all_neurons) > max_neurons:
        # Prioritize: gained, lost, then winners
        prioritized = list(gained) + list(lost) + [w for w in winners if w not in gained]
        all_neurons = set(prioritized[:max_neurons])

    # Layout in grid
    import math
    cols = max(1, int(math.sqrt(len(all_neurons))))

    for i, neuron_id in enumerate(sorted(all_neurons)):
        row, col = divmod(i, cols)
        x = 30 + col * 20
        y = 30 + row * 20

        classes = ["neuron"]
        if neuron_id in gained:
            classes.append("neuron-gained")
        elif neuron_id in lost:
            classes.append("neuron-lost")
        elif neuron_id in set(winners):
            classes.append("neuron-winner")

        elements.append({
            "data": {
                "id": f"n_{neuron_id}",
                "label": str(neuron_id),
                "neuron_id": neuron_id,
            },
            "position": {"x": x, "y": y},
            "classes": " ".join(classes),
        })

    return elements


# ── Component builders ───────────────────────────────────────────────────────

def create_graph_component(
    elements: list[dict],
    graph_id: str = "cyto-graph",
    height: str = "300px",
    layout_name: str = "preset",
) -> cyto.Cytoscape:
    """Create a Cytoscape component with the standard stylesheet."""
    return cyto.Cytoscape(
        id=graph_id,
        elements=elements,
        stylesheet=GRAPH_STYLESHEET,
        style={"width": "100%", "height": height,
               "backgroundColor": "#0f0f1a"},
        layout={"name": layout_name},
        userZoomingEnabled=True,
        userPanningEnabled=True,
        boxSelectionEnabled=False,
    )


def create_graph_panel(
    step: dict,
    step_idx: int,
    selected_area: str = None,
    level: int = 0,
    area_names: list[str] = None,
    assembly_tracker=None,
) -> html.Div:
    """Create the complete graph panel with level info and component."""
    level_labels = {
        0: "Area Graph — click an area to drill down",
        1: f"Assembly View: {selected_area or '?'}",
        2: f"Neuron View: {selected_area or '?'}",
    }

    if level == 0:
        elements = build_area_graph(step, area_names)
        height = "300px"
    elif level == 1 and selected_area:
        elements = build_assembly_view(
            selected_area, step, assembly_tracker, step_idx
        )
        height = "300px"
    elif level == 2 and selected_area:
        elements = build_neuron_view(selected_area, step)
        height = "350px"
    else:
        elements = []
        height = "200px"

    return html.Div([
        html.Div(
            level_labels.get(level, ""),
            style={"color": "#888", "fontSize": "12px", "marginBottom": "8px"},
        ),
        create_graph_component(elements, height=height),
    ])
