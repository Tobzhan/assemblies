"""
SimulationLogger — Captures per-step brain state for visualization.

Logs winners, attribution, template overlaps, activation margins,
and diffs to a single JSONL file.
"""

import json
import os
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class InputStats:
    """Summary statistics for neuron input distribution in an area."""
    min: float = 0.0
    max: float = 0.0
    mean: float = 0.0
    std: float = 0.0
    median: float = 0.0
    p25: float = 0.0
    p75: float = 0.0

    @staticmethod
    def from_array(arr):
        """Build InputStats from a numpy array of neuron inputs."""
        if arr is None or len(arr) == 0:
            return InputStats()
        a = np.asarray(arr, dtype=np.float64)
        return InputStats(
            min=float(np.min(a)),
            max=float(np.max(a)),
            mean=float(np.mean(a)),
            std=float(np.std(a)),
            median=float(np.median(a)),
            p25=float(np.percentile(a, 25)),
            p75=float(np.percentile(a, 75)),
        )


@dataclass
class AreaActivation:
    """Activation snapshot for a single area at one timestep."""
    area: str
    winners: list                     # neuron IDs (sorted)
    winner_inputs: list               # input score per winner
    num_new_winners: int = 0
    activation_margin: float = 0.0    # gap k-th vs (k+1)-th

    # Attribution
    attribution: dict = field(default_factory=dict)   # source → total_input

    # Template match
    template_name: Optional[str] = None
    template_overlap: float = 0.0     # Jaccard with template

    # Diff from previous step
    winners_gained: list = field(default_factory=list)
    winners_lost: list = field(default_factory=list)
    overlap_with_prev: float = 0.0

    # Summary stats
    input_stats: Optional[InputStats] = None


@dataclass
class StepLog:
    """One logged timestep."""
    step: int
    phase: str = "project"

    # Per-area activations (only targeted areas)
    activations: dict = field(default_factory=dict)   # area_name → AreaActivation

    # DFA-specific
    dfa_state: Optional[str] = None
    dfa_symbol: Optional[str] = None
    dfa_transition: Optional[str] = None


# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------

@dataclass
class AssemblyTemplate:
    """A known / expected assembly for comparison."""
    name: str
    area: str
    neuron_ids: set           # expected winner neuron IDs


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class SimulationLogger:
    """Captures per-step brain state and writes to JSONL."""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        self.log_path = os.path.join(log_dir, "simulation_log.jsonl")
        self.steps: list[StepLog] = []
        self.templates: dict[str, AssemblyTemplate] = {}   # name → template
        self._prev_winners: dict[str, set] = {}            # area → set(winners)
        self._step_counter = 0

    # -- Templates -----------------------------------------------------------

    def register_template(self, name: str, area: str, neuron_ids):
        """Register a known assembly for overlap tracking."""
        self.templates[name] = AssemblyTemplate(
            name=name, area=area, neuron_ids=set(neuron_ids)
        )

    # -- Activation recording ------------------------------------------------

    def record_activation(
        self,
        area: str,
        winners: list,
        winner_inputs: list | None = None,
        all_inputs: np.ndarray | None = None,
        num_new_winners: int = 0,
        attribution: dict | None = None,
    ) -> AreaActivation:
        """Build an AreaActivation, computing overlaps & margins."""
        winners_set = set(winners)

        # Template overlap
        tmpl_name = None
        tmpl_overlap = 0.0
        for t in self.templates.values():
            if t.area == area:
                inter = len(winners_set & t.neuron_ids)
                union = len(winners_set | t.neuron_ids)
                overlap = inter / union if union > 0 else 0.0
                if overlap > tmpl_overlap:
                    tmpl_overlap = overlap
                    tmpl_name = t.name

        # Diff from previous step
        prev = self._prev_winners.get(area, set())
        gained = sorted(winners_set - prev)
        lost = sorted(prev - winners_set)
        inter_prev = len(winners_set & prev)
        union_prev = len(winners_set | prev)
        overlap_prev = inter_prev / union_prev if union_prev > 0 else 0.0

        # Activation margin
        margin = 0.0
        if all_inputs is not None and len(all_inputs) > len(winners):
            sorted_inputs = np.sort(all_inputs)[::-1]
            k = len(winners)
            if k < len(sorted_inputs):
                margin = float(sorted_inputs[k - 1] - sorted_inputs[k])

        # Input stats
        input_stats = None
        if all_inputs is not None:
            input_stats = InputStats.from_array(all_inputs)

        if winner_inputs is None:
            winner_inputs = []

        act = AreaActivation(
            area=area,
            winners=sorted(winners),
            winner_inputs=[float(x) for x in winner_inputs],
            num_new_winners=num_new_winners,
            activation_margin=margin,
            attribution=attribution or {},
            template_name=tmpl_name,
            template_overlap=tmpl_overlap,
            winners_gained=gained,
            winners_lost=lost,
            overlap_with_prev=overlap_prev,
            input_stats=input_stats,
        )
        # Update previous-winners cache
        self._prev_winners[area] = winners_set
        return act

    # -- Step logging --------------------------------------------------------

    def log_step(
        self,
        activations: dict[str, AreaActivation],
        phase: str = "project",
        dfa_state: str | None = None,
        dfa_symbol: str | None = None,
        dfa_transition: str | None = None,
    ) -> StepLog:
        """Record a full timestep."""
        step = StepLog(
            step=self._step_counter,
            phase=phase,
            activations=activations,
            dfa_state=dfa_state,
            dfa_symbol=dfa_symbol,
            dfa_transition=dfa_transition,
        )
        self.steps.append(step)
        self._step_counter += 1
        return step

    # -- Serialization -------------------------------------------------------

    def _serialize_step(self, step: StepLog) -> dict:
        """Convert a StepLog to a JSON-serializable dict."""
        d = {
            "step": step.step,
            "phase": step.phase,
            "dfa_state": step.dfa_state,
            "dfa_symbol": step.dfa_symbol,
            "dfa_transition": step.dfa_transition,
            "activations": {},
        }
        for area_name, act in step.activations.items():
            ad = asdict(act)
            # InputStats is nested — convert to dict if present
            if ad.get("input_stats") and isinstance(ad["input_stats"], dict):
                pass  # already a dict via asdict
            d["activations"][area_name] = ad
        return d

    def save(self):
        """Write all steps to JSONL file."""
        os.makedirs(self.log_dir, exist_ok=True)
        with open(self.log_path, "w", encoding="utf-8") as f:
            for step in self.steps:
                line = json.dumps(self._serialize_step(step), separators=(",", ":"))
                f.write(line + "\n")

    def load(self, path: str | None = None) -> list[dict]:
        """Load a JSONL log file and return list of step dicts."""
        p = path or self.log_path
        rows = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
