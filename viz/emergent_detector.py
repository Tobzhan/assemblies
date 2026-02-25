"""
Emergent Assembly Detector — Detects assemblies from winner history
using sliding-window overlap analysis.

For non-DFA simulations where assembly membership is not known in advance,
this module detects when winners converge into a stable pattern.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EmergentAssembly:
    """An assembly detected from winner convergence."""
    id: str
    area: str
    canonical_neurons: set
    first_detected: int       # step when overlap first exceeded threshold
    last_updated: int
    converged_at: Optional[int] = None  # step when fully converged
    status: str = "forming"   # forming | converged | drifting | split | lost
    overlap_history: list = field(default_factory=list)
    member_count_history: list = field(default_factory=list)


class EmergentDetector:
    """Detects emergent assemblies from winner history using overlap analysis."""

    def __init__(
        self,
        convergence_threshold: float = 0.8,
        convergence_window: int = 5,
        split_threshold: float = 0.4,
    ):
        self.convergence_threshold = convergence_threshold
        self.convergence_window = convergence_window
        self.split_threshold = split_threshold

        self.assemblies: dict[str, EmergentAssembly] = {}
        self._next_id: dict[str, int] = {}
        self._winner_history: dict[str, list[set]] = {}  # area → list of winner sets

    def _make_id(self, area: str) -> str:
        n = self._next_id.get(area, 0)
        self._next_id[area] = n + 1
        return f"{area}:asm_{n}"

    def _jaccard(self, a: set, b: set) -> float:
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def observe(self, area: str, winners: list, step: int) -> Optional[str]:
        """Observe winners for an area at a given step.
        Returns assembly ID if one is detected/maintained, None otherwise."""
        winners_set = set(winners)

        # Track history
        if area not in self._winner_history:
            self._winner_history[area] = []
        self._winner_history[area].append(winners_set)

        if not winners_set:
            return None

        # Calculate overlap with previous step
        history = self._winner_history[area]
        if len(history) < 2:
            return None

        overlap_prev = self._jaccard(history[-1], history[-2])

        # Find best matching existing assembly
        best_asm = None
        best_overlap = 0.0
        for asm in self.assemblies.values():
            if asm.area == area and asm.status != "lost":
                ov = self._jaccard(winners_set, asm.canonical_neurons)
                if ov > best_overlap:
                    best_overlap = ov
                    best_asm = asm

        if best_asm and best_overlap >= self.split_threshold:
            # Update existing assembly
            best_asm.overlap_history.append(best_overlap)
            best_asm.member_count_history.append(len(winners_set))
            best_asm.last_updated = step

            # Check convergence window
            recent = best_asm.overlap_history[-self.convergence_window:]
            if (len(recent) >= self.convergence_window
                    and all(o >= self.convergence_threshold for o in recent)):
                if best_asm.status != "converged":
                    best_asm.converged_at = step
                best_asm.status = "converged"
                best_asm.canonical_neurons = winners_set
            elif best_overlap >= self.convergence_threshold:
                best_asm.canonical_neurons = winners_set
                if best_asm.status == "converged":
                    pass  # still converged
                else:
                    best_asm.status = "forming"
            elif best_overlap >= self.split_threshold:
                if best_asm.status == "converged":
                    best_asm.status = "drifting"
            else:
                best_asm.status = "lost"

            return best_asm.id

        elif overlap_prev >= self.convergence_threshold:
            # New assembly forming
            asm_id = self._make_id(area)
            asm = EmergentAssembly(
                id=asm_id,
                area=area,
                canonical_neurons=winners_set,
                first_detected=step,
                last_updated=step,
                overlap_history=[overlap_prev],
                member_count_history=[len(winners_set)],
            )
            self.assemblies[asm_id] = asm
            return asm_id

        return None

    def process_log(self, steps: list[dict]) -> dict[int, dict[str, str]]:
        """Process all steps and return step→{area→assembly_id} mapping."""
        mapping = {}
        for step in steps:
            step_idx = step.get("step", 0)
            step_map = {}
            for area, act in step.get("activations", {}).items():
                winners = act.get("winners", [])
                asm_id = self.observe(area, winners, step_idx)
                if asm_id:
                    step_map[area] = asm_id
            mapping[step_idx] = step_map
        return mapping

    def get_convergence_series(self, area: str) -> list[float]:
        """Get step-by-step overlap series for an area (between consecutive steps)."""
        history = self._winner_history.get(area, [])
        if len(history) < 2:
            return [0.0] * len(history)

        series = [0.0]  # first step has no previous
        for i in range(1, len(history)):
            series.append(self._jaccard(history[i], history[i - 1]))
        return series

    def get_detected_assemblies(self, area: str = None) -> list[EmergentAssembly]:
        """Get all detected assemblies, optionally filtered by area."""
        return [
            a for a in self.assemblies.values()
            if (area is None or a.area == area) and a.status != "lost"
        ]

    def get_summary(self) -> list[dict]:
        """Get a summary for display."""
        return [
            {
                "id": a.id,
                "area": a.area,
                "status": a.status,
                "first_detected": a.first_detected,
                "converged_at": a.converged_at,
                "canonical_size": len(a.canonical_neurons),
                "total_observations": len(a.overlap_history),
            }
            for a in self.assemblies.values()
        ]
