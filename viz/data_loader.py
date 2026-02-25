"""
data_loader.py — Loads simulation JSONL logs into structured Python objects
for the Dash visualization app.
"""

import json
import os


class SimulationData:
    """Parsed simulation log — provides step-by-step access for the UI."""

    def __init__(self, steps: list[dict]):
        self.steps = steps
        self.num_steps = len(steps)

    # -- Access helpers -------------------------------------------------------

    def get_step(self, idx: int) -> dict:
        """Get a step by index (0-based)."""
        if 0 <= idx < self.num_steps:
            return self.steps[idx]
        return {}

    def get_area_names(self) -> list[str]:
        """Get all area names that appear in the log."""
        names = set()
        for step in self.steps:
            for area in step.get("activations", {}):
                names.add(area)
        return sorted(names)

    def get_dfa_path(self) -> list[dict]:
        """Extract the DFA state path: list of {step, state, symbol, transition}."""
        path = []
        for step in self.steps:
            path.append({
                "step": step.get("step", 0),
                "state": step.get("dfa_state"),
                "symbol": step.get("dfa_symbol"),
                "transition": step.get("dfa_transition"),
            })
        return path

    def get_area_activation(self, step_idx: int, area: str) -> dict:
        """Get activation data for a specific area at a specific step."""
        step = self.get_step(step_idx)
        return step.get("activations", {}).get(area, {})

    def get_overlap_series(self, area: str) -> list[float]:
        """Get template overlap over time for a given area."""
        series = []
        for step in self.steps:
            act = step.get("activations", {}).get(area, {})
            series.append(act.get("template_overlap", 0.0))
        return series

    def get_margin_series(self, area: str) -> list[float]:
        """Get activation margin over time for a given area."""
        series = []
        for step in self.steps:
            act = step.get("activations", {}).get(area, {})
            series.append(act.get("activation_margin", 0.0))
        return series

    def get_winner_count_series(self, area: str) -> list[int]:
        """Get number of winners per step for an area."""
        series = []
        for step in self.steps:
            act = step.get("activations", {}).get(area, {})
            winners = act.get("winners", [])
            series.append(len(winners))
        return series

    # -- Brain thought (narrative) -------------------------------------------

    def generate_thought(self, step_idx: int) -> str:
        """Generate a human-readable narrative for a given step."""
        step = self.get_step(step_idx)
        if not step:
            return "No data for this step."

        lines = []
        phase = step.get("phase", "unknown")
        dfa_state = step.get("dfa_state")
        dfa_symbol = step.get("dfa_symbol")
        dfa_transition = step.get("dfa_transition")

        lines.append(f"🧠 Brain Thought — Step {step.get('step', '?')}")
        lines.append("")

        if phase == "init":
            lines.append(f"PHASE: Initialization")
            lines.append(f"Starting state: {dfa_state}")
        else:
            lines.append(f"PHASE: Projection")
            if dfa_symbol:
                lines.append(f"INPUT: Symbol '{dfa_symbol}' fired")
            if dfa_transition:
                lines.append(f"TRANSITION: {dfa_transition}")

        lines.append("")
        lines.append("AREA ACTIVATIONS:")

        for area_name, act in step.get("activations", {}).items():
            winners = act.get("winners", [])
            n_winners = len(winners)
            template = act.get("template_name", None)
            overlap = act.get("template_overlap", 0)
            margin = act.get("activation_margin", 0)
            new_w = act.get("num_new_winners", 0)

            if n_winners == 0:
                status = "dormant"
                bar = ""
            else:
                pct = overlap * 100
                bar_len = int(overlap * 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                status = f"{n_winners} winners"

            line = f"  {area_name}: {status}"
            if template:
                line += f" | template={template} overlap={overlap:.0%}"
            if margin > 0:
                line += f" | margin={margin:.1f}"
            if new_w > 0:
                line += f" | {new_w} NEW"
            lines.append(line)

            # Attribution
            attr = act.get("attribution", {})
            if attr:
                for src, val in attr.items():
                    if val > 0:
                        lines.append(f"    ← {src}: {val:.1f}")

        lines.append("")
        if dfa_state:
            lines.append(f"RESULT: Current state = {dfa_state}")

        return "\n".join(lines)

    # -- Diff between steps --------------------------------------------------

    def generate_diff(self, step_idx: int) -> str:
        """Generate a diff summary between step N-1 and step N."""
        step = self.get_step(step_idx)
        if not step:
            return "No data."

        lines = [f"📊 Diff: step {step_idx - 1} → step {step_idx}", ""]

        for area_name, act in step.get("activations", {}).items():
            gained = act.get("winners_gained", [])
            lost = act.get("winners_lost", [])
            overlap = act.get("overlap_with_prev", 0)

            if not gained and not lost:
                lines.append(f"  {area_name}: no change (overlap={overlap:.0%})")
            else:
                lines.append(
                    f"  {area_name}: +{len(gained)} gained, "
                    f"-{len(lost)} lost (overlap={overlap:.0%})"
                )

        return "\n".join(lines)


def load_simulation(path: str) -> SimulationData:
    """Load a JSONL simulation log file."""
    steps = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                steps.append(json.loads(line))
    return SimulationData(steps)
