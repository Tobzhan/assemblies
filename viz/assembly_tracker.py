"""
Assembly Tracker — Assigns stable IDs to assemblies across timesteps.

Tracks convergence, drift, splits, and merges using Jaccard overlap
against canonical neuron sets.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrackedAssembly:
    """A tracked assembly with a stable ID."""
    id: str
    area: str
    canonical_neurons: set          # rolling canonical neuron set
    first_seen: int                 # step index
    last_seen: int                  # step index
    status: str = "forming"         # forming | stable | drifting | lost
    overlap_history: list = field(default_factory=list)  # overlap per step
    consecutive_stable: int = 0     # consecutive steps with overlap > threshold


class AssemblyTracker:
    """Tracks assemblies across timesteps with stable IDs."""

    def __init__(
        self,
        convergence_threshold: float = 0.8,
        convergence_steps: int = 5,
        persistence_threshold: float = 0.6,
        drift_threshold: float = 0.4,
    ):
        self.convergence_threshold = convergence_threshold
        self.convergence_steps = convergence_steps
        self.persistence_threshold = persistence_threshold
        self.drift_threshold = drift_threshold

        self.assemblies: dict[str, TrackedAssembly] = {}
        self._next_id: dict[str, int] = {}  # per-area auto-increment
        self._prev_winners: dict[str, set] = {}  # area → previous winners

    def _make_id(self, area: str) -> str:
        """Generate a new unique assembly ID for an area."""
        n = self._next_id.get(area, 0)
        self._next_id[area] = n + 1
        return f"{area}:{n}"

    def _jaccard(self, a: set, b: set) -> float:
        """Jaccard similarity between two sets."""
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def update(self, area: str, winners: list, step: int) -> Optional[str]:
        """Update tracking for an area at a given step.
        Returns the assembly ID if one is assigned, None otherwise."""
        winners_set = set(winners)

        if not winners_set:
            # No winners → no assembly
            self._prev_winners[area] = set()
            # Mark any active assembly as lost
            for asm in self.assemblies.values():
                if asm.area == area and asm.status in ("stable", "forming", "drifting"):
                    asm.status = "lost"
                    asm.last_seen = step
            return None

        prev = self._prev_winners.get(area, set())
        overlap_with_prev = self._jaccard(winners_set, prev)
        self._prev_winners[area] = winners_set

        # Find best matching existing assembly for this area
        best_asm = None
        best_overlap = 0.0
        for asm in self.assemblies.values():
            if asm.area == area and asm.status != "lost":
                ov = self._jaccard(winners_set, asm.canonical_neurons)
                if ov > best_overlap:
                    best_overlap = ov
                    best_asm = asm

        if best_asm and best_overlap >= self.persistence_threshold:
            # Continue tracking this assembly
            best_asm.last_seen = step
            best_asm.overlap_history.append(best_overlap)

            # Update canonical neurons (rolling update: merge with new)
            if best_overlap >= self.convergence_threshold:
                best_asm.canonical_neurons = winners_set
                best_asm.consecutive_stable += 1
            else:
                best_asm.consecutive_stable = 0

            # Update status
            if best_overlap >= self.convergence_threshold:
                if best_asm.consecutive_stable >= self.convergence_steps:
                    best_asm.status = "stable"
                elif best_asm.status not in ("stable",):
                    best_asm.status = "forming"
            elif best_overlap >= self.persistence_threshold:
                # Below convergence but above persistence → drifting
                if best_asm.status in ("stable", "forming"):
                    best_asm.status = "drifting"
            elif best_overlap >= self.drift_threshold:
                best_asm.status = "drifting"
            else:
                best_asm.status = "lost"

            return best_asm.id

        elif best_asm and best_overlap >= self.drift_threshold:
            # Drifting — keep the ID but mark as drifting
            best_asm.last_seen = step
            best_asm.overlap_history.append(best_overlap)
            best_asm.status = "drifting"
            best_asm.consecutive_stable = 0
            return best_asm.id

        else:
            # New assembly forming
            new_id = self._make_id(area)
            asm = TrackedAssembly(
                id=new_id,
                area=area,
                canonical_neurons=winners_set,
                first_seen=step,
                last_seen=step,
                status="forming",
                overlap_history=[overlap_with_prev],
                consecutive_stable=1 if overlap_with_prev >= self.convergence_threshold else 0,
            )
            self.assemblies[new_id] = asm
            return new_id

    def get_assembly(self, asm_id: str) -> Optional[TrackedAssembly]:
        """Get a tracked assembly by ID."""
        return self.assemblies.get(asm_id)

    def get_active_assemblies(self, area: str = None) -> list[TrackedAssembly]:
        """Get all non-lost assemblies, optionally filtered by area."""
        result = []
        for asm in self.assemblies.values():
            if asm.status != "lost":
                if area is None or asm.area == area:
                    result.append(asm)
        return result

    def get_area_history(self, area: str) -> list[TrackedAssembly]:
        """Get full history (including lost) for an area."""
        return [a for a in self.assemblies.values() if a.area == area]

    def process_log(self, steps: list[dict]) -> dict[int, dict[str, str]]:
        """Process an entire log and return step→{area→assembly_id} mapping."""
        mapping = {}
        for step in steps:
            step_idx = step.get("step", 0)
            step_mapping = {}
            for area_name, act in step.get("activations", {}).items():
                winners = act.get("winners", [])
                asm_id = self.update(area_name, winners, step_idx)
                if asm_id:
                    step_mapping[area_name] = asm_id
            mapping[step_idx] = step_mapping
        return mapping

    def get_summary(self) -> list[dict]:
        """Get a summary of all tracked assemblies."""
        return [
            {
                "id": a.id,
                "area": a.area,
                "status": a.status,
                "first_seen": a.first_seen,
                "last_seen": a.last_seen,
                "canonical_size": len(a.canonical_neurons),
                "history_length": len(a.overlap_history),
                "consecutive_stable": a.consecutive_stable,
            }
            for a in self.assemblies.values()
        ]
