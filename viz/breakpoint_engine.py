"""
Breakpoint Engine — Conditional breakpoints for simulation replay.

Define conditions (overlap < X, margin < Y, new_winners > Z) and the engine
scans the log to find steps where those conditions trigger.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


class ConditionType(str, Enum):
    """Supported breakpoint condition types."""
    OVERLAP_BELOW = "overlap_below"       # template_overlap < threshold
    MARGIN_BELOW = "margin_below"         # activation_margin < threshold
    NEW_WINNERS_ABOVE = "new_winners_above"  # num_new_winners > threshold
    OVERLAP_PREV_BELOW = "overlap_prev_below"  # overlap_with_prev < threshold
    WINNERS_CHANGED = "winners_changed"   # any winner gained or lost
    STATE_CHANGED = "state_changed"       # DFA state changed from prev step


@dataclass
class Breakpoint:
    """A single breakpoint condition."""
    id: str
    condition: ConditionType
    threshold: float = 0.0
    area: Optional[str] = None  # None = any area
    enabled: bool = True

    def to_dict(self) -> dict:
        d = asdict(self)
        d["condition"] = self.condition.value
        return d

    @staticmethod
    def from_dict(d: dict) -> "Breakpoint":
        return Breakpoint(
            id=d["id"],
            condition=ConditionType(d["condition"]),
            threshold=d.get("threshold", 0.0),
            area=d.get("area"),
            enabled=d.get("enabled", True),
        )


@dataclass
class BreakpointHit:
    """Records a breakpoint firing at a specific step."""
    breakpoint_id: str
    step: int
    area: str
    actual_value: float
    message: str


class BreakpointEngine:
    """Evaluates breakpoints against simulation steps."""

    def __init__(self):
        self.breakpoints: dict[str, Breakpoint] = {}
        self._next_id = 0

    def add_breakpoint(
        self,
        condition: ConditionType,
        threshold: float = 0.0,
        area: str | None = None,
        bp_id: str | None = None,
    ) -> Breakpoint:
        """Add a new breakpoint. Returns the created Breakpoint."""
        if bp_id is None:
            bp_id = f"bp_{self._next_id}"
            self._next_id += 1
        bp = Breakpoint(
            id=bp_id,
            condition=condition,
            threshold=threshold,
            area=area,
        )
        self.breakpoints[bp_id] = bp
        return bp

    def remove_breakpoint(self, bp_id: str) -> bool:
        """Remove a breakpoint by ID. Returns True if removed."""
        if bp_id in self.breakpoints:
            del self.breakpoints[bp_id]
            return True
        return False

    def toggle_breakpoint(self, bp_id: str) -> bool:
        """Toggle a breakpoint's enabled state. Returns new state."""
        if bp_id in self.breakpoints:
            self.breakpoints[bp_id].enabled = not self.breakpoints[bp_id].enabled
            return self.breakpoints[bp_id].enabled
        return False

    def clear_all(self):
        """Remove all breakpoints."""
        self.breakpoints.clear()
        self._next_id = 0

    def evaluate_step(
        self, step: dict, prev_step: dict | None = None
    ) -> list[BreakpointHit]:
        """Check all enabled breakpoints against a single step.
        Returns list of BreakpointHit for each triggered condition."""
        hits = []
        for bp in self.breakpoints.values():
            if not bp.enabled:
                continue
            step_hits = self._check_breakpoint(bp, step, prev_step)
            hits.extend(step_hits)
        return hits

    def _check_breakpoint(
        self, bp: Breakpoint, step: dict, prev_step: dict | None
    ) -> list[BreakpointHit]:
        """Check a single breakpoint against a step."""
        hits = []
        step_idx = step.get("step", 0)

        # For state_changed, compare DFA states
        if bp.condition == ConditionType.STATE_CHANGED:
            if prev_step is not None:
                prev_state = prev_step.get("dfa_state")
                cur_state = step.get("dfa_state")
                if prev_state and cur_state and prev_state != cur_state:
                    hits.append(BreakpointHit(
                        breakpoint_id=bp.id,
                        step=step_idx,
                        area="DFA",
                        actual_value=0,
                        message=f"State changed: {prev_state} → {cur_state}",
                    ))
            return hits

        # For area-based conditions
        activations = step.get("activations", {})
        areas_to_check = [bp.area] if bp.area else list(activations.keys())

        for area_name in areas_to_check:
            act = activations.get(area_name, {})
            if not act:
                continue

            value = None
            triggered = False
            message = ""

            if bp.condition == ConditionType.OVERLAP_BELOW:
                value = act.get("template_overlap", 1.0)
                triggered = value < bp.threshold
                message = (f"Template overlap {value:.1%} < {bp.threshold:.1%} "
                          f"in {area_name}")

            elif bp.condition == ConditionType.MARGIN_BELOW:
                value = act.get("activation_margin", float("inf"))
                triggered = value < bp.threshold
                message = (f"Activation margin {value:.2f} < {bp.threshold:.2f} "
                          f"in {area_name}")

            elif bp.condition == ConditionType.NEW_WINNERS_ABOVE:
                value = float(act.get("num_new_winners", 0))
                triggered = value > bp.threshold
                message = (f"New winners {int(value)} > {int(bp.threshold)} "
                          f"in {area_name}")

            elif bp.condition == ConditionType.OVERLAP_PREV_BELOW:
                value = act.get("overlap_with_prev", 1.0)
                triggered = value < bp.threshold
                message = (f"Overlap with prev {value:.1%} < {bp.threshold:.1%} "
                          f"in {area_name}")

            elif bp.condition == ConditionType.WINNERS_CHANGED:
                gained = act.get("winners_gained", [])
                lost = act.get("winners_lost", [])
                n_changed = len(gained) + len(lost)
                value = float(n_changed)
                triggered = n_changed > 0
                message = (f"Winners changed: +{len(gained)} -{len(lost)} "
                          f"in {area_name}")

            if triggered and value is not None:
                hits.append(BreakpointHit(
                    breakpoint_id=bp.id,
                    step=step_idx,
                    area=area_name,
                    actual_value=value,
                    message=message,
                ))

        return hits

    def find_next_breakpoint(
        self, steps: list[dict], from_step: int = 0
    ) -> BreakpointHit | None:
        """Scan forward from from_step to find the next triggered breakpoint.
        Returns the first BreakpointHit found, or None."""
        for i in range(from_step, len(steps)):
            prev = steps[i - 1] if i > 0 else None
            hits = self.evaluate_step(steps[i], prev)
            if hits:
                return hits[0]
        return None

    def find_all_breakpoints(
        self, steps: list[dict]
    ) -> list[BreakpointHit]:
        """Scan all steps and return all breakpoint hits."""
        all_hits = []
        for i, step in enumerate(steps):
            prev = steps[i - 1] if i > 0 else None
            hits = self.evaluate_step(step, prev)
            all_hits.extend(hits)
        return all_hits

    def get_breakpoint_summary(self) -> list[dict]:
        """Get a summary of all breakpoints for display."""
        return [bp.to_dict() for bp in self.breakpoints.values()]
