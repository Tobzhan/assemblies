"""
LiveBrain — Wraps brain.py's Brain class with operation logging
and state export for real-time visualization.

Each operation (add_area, stimulate, project) is logged as a step
with before/after snapshots of winners, overlaps, and weight stats.
"""

import os
import sys
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain import Brain, Area


@dataclass
class StepRecord:
    """A single recorded operation step."""
    step: int
    operation: str          # "add_area", "add_stimulus", "stimulate", "project"
    description: str        # human-readable, e.g. "project stim_A → A"
    areas_snapshot: dict    # area_name → {winners, n_winners, overlap_prev, ...}
    fibers: list            # [(src, dst), ...] active fibers
    learning: bool          # whether plasticity was on


class LiveBrain:
    """Interactive brain wrapper with operation logging."""

    def __init__(self, p=0.1, beta=0.05, seed=0):
        self.brain = Brain(p, save_size=True, save_winners=True, seed=seed)
        self.default_beta = beta
        self.learning = True
        self.steps: list[StepRecord] = []
        self._step_counter = 0
        self._prev_winners: dict[str, set] = {}  # area → previous winners set
        self._fibers: list[tuple[str, str]] = []  # (src, dst) pairs
        self._last_snapshot: dict = {}  # cached snapshot from last _log()
        self._p = p
        self._seed = seed

    def reset(self):
        """Reset brain to fresh state, clearing all areas, stimuli, and history."""
        self.brain = Brain(self._p, save_size=True, save_winners=True, seed=self._seed)
        self.learning = True
        self.steps.clear()
        self._step_counter = 0
        self._prev_winners.clear()
        self._fibers.clear()
        self._last_snapshot.clear()

    # ── Area & Stimulus management ──────────────────────────────────────

    def add_area(self, name: str, n: int = 1000, k: int = 100,
                 beta: float = None) -> StepRecord:
        """Add an explicit brain area."""
        beta = beta if beta is not None else self.default_beta
        self.brain.add_explicit_area(name, n, k, beta)
        self._prev_winners[name] = set()

        # Auto-create fibers to/from existing areas
        for other in list(self.brain.area_by_name.keys()):
            if other != name:
                if (name, other) not in self._fibers:
                    self._fibers.append((name, other))
                if (other, name) not in self._fibers:
                    self._fibers.append((other, name))

        return self._log("add_area", f"Added area '{name}' (n={n}, k={k}, β={beta})",
                         involved_areas={name})

    def add_stimulus(self, name: str, size: int = 100) -> StepRecord:
        """Add a stimulus source."""
        self.brain.add_stimulus(name, size)
        return self._log("add_stimulus", f"Added stimulus '{name}' (size={size})",
                         involved_areas=set())

    def get_areas(self) -> list[str]:
        """Get list of area names."""
        return list(self.brain.area_by_name.keys())

    def get_stimuli(self) -> list[str]:
        """Get list of stimulus names."""
        return list(self.brain.stimulus_size_by_name.keys())

    def get_fibers(self) -> list[tuple[str, str]]:
        """Get list of (src, dst) fiber pairs."""
        return list(self._fibers)

    # ── Operations ──────────────────────────────────────────────────────

    def stimulate(self, stimulus: str, area: str, rounds: int = 1) -> StepRecord:
        """Apply a stimulus to an area. Logs each round individually."""
        record = None
        for r in range(rounds):
            self.brain.project(
                {stimulus: [area]},   # stim → target areas
                {},                    # no area→area projection
            )
            record = self._log("stimulate",
                               f"Stimulate '{stimulus}' → '{area}' [{r+1}/{rounds}]",
                               involved_areas={area})
        return record

    def project(self, src: str, dst: str, rounds: int = 1) -> StepRecord:
        """Project from src area to dst area. Logs each round individually."""
        record = None
        for r in range(rounds):
            self.brain.project(
                {},                     # no stimuli
                {src: [dst]},           # src → dst
            )
            record = self._log("project",
                               f"Project '{src}' → '{dst}' [{r+1}/{rounds}]",
                               involved_areas={dst})
        return record

    def reciprocal_project(self, area_a: str, area_b: str,
                           rounds: int = 1) -> StepRecord:
        """Reciprocal projection between two areas. Logs each round."""
        record = None
        for r in range(rounds):
            self.brain.project(
                {},
                {area_a: [area_b], area_b: [area_a]},
            )
            record = self._log("reciprocal",
                               f"Reciprocal '{area_a}' ↔ '{area_b}' [{r+1}/{rounds}]",
                               involved_areas={area_a, area_b})
        return record

    def stimulate_and_project(self, stimulus: str, stim_area: str,
                              src: str, dst: str,
                              rounds: int = 1) -> StepRecord:
        """Stimulate + project in one step (common pattern)."""
        for _ in range(rounds):
            stim_map = {stimulus: [stim_area]}
            area_map = {src: [dst]} if src != stim_area else {}
            # If stim_area == src, just stimulate into dst
            if src == stim_area:
                area_map = {stim_area: [dst]}
            self.brain.project(stim_map, area_map)
        return self._log("stim+project",
                         f"Stim '{stimulus}'→'{stim_area}' + Project '{src}'→'{dst}' ×{rounds}",
                         involved_areas={stim_area, dst})

    def associate(self, stim_a: str, stim_b: str, area: str,
                  rounds: int = 1) -> StepRecord:
        """Associate: fire two stimuli into the same area simultaneously.

        This is the AC 'associate' operation — over repeated rounds,
        the area forms a single assembly that responds to both stimuli.
        """
        record = None
        for r in range(rounds):
            self.brain.project(
                {stim_a: [area], stim_b: [area]},  # both stimuli → same area
                {},
            )
            record = self._log("associate",
                               f"Associate '{stim_a}'+'{stim_b}' → '{area}' [{r+1}/{rounds}]",
                               involved_areas={area})
        return record

    def merge(self, src_a: str, src_b: str, dst: str,
              rounds: int = 1) -> StepRecord:
        """Merge: project two source areas into a target area simultaneously.

        This is the AC 'merge' operation — creates a combined assembly
        in dst that represents inputs from both src_a and src_b.
        """
        record = None
        for r in range(rounds):
            self.brain.project(
                {},
                {src_a: [dst], src_b: [dst]},  # both areas → target
            )
            record = self._log("merge",
                               f"Merge '{src_a}'+'{src_b}' → '{dst}' [{r+1}/{rounds}]",
                               involved_areas={dst})
        return record

    # ── Learning toggle ─────────────────────────────────────────────────

    def set_learning(self, on: bool) -> None:
        """Toggle Hebbian plasticity. on=True → learning, on=False → β=0."""
        self.learning = on
        self.brain.disable_plasticity = not on

    # ── State export ────────────────────────────────────────────────────

    def get_area_state(self, area_name: str) -> dict:
        """Get current state of a specific area."""
        area = self.brain.area_by_name.get(area_name)
        if not area:
            return {}

        winners = set(area.winners) if area.winners else set()
        prev = self._prev_winners.get(area_name, set())

        # Compute overlap with previous
        if winners and prev:
            jaccard = len(winners & prev) / len(winners | prev)
        else:
            # No winners or no prev → 0% overlap (nothing to compare)
            jaccard = 0.0

        gained = winners - prev
        lost = prev - winners

        return {
            "name": area_name,
            "n": area.n,
            "k": area.k,
            "beta": area.beta,
            "n_winners": len(winners),
            "winners": sorted(winners)[:50],  # cap for display
            "overlap_prev": jaccard,
            "gained": len(gained),
            "lost": len(lost),
            "gained_set": gained,     # actual neuron IDs
            "lost_set": lost,         # actual neuron IDs
            "winners_set": winners,   # full set of winner IDs
            "num_ever_fired": area.get_num_ever_fired(),
            "w": area.w,
        }

    def get_full_state(self) -> dict:
        """Get snapshot of all areas."""
        return {
            name: self.get_area_state(name)
            for name in self.brain.area_by_name
        }

    def _get_connectome(self, from_name: str, to_area: str):
        """Resolve connectome: area→area or stimulus→area."""
        # Try area→area first
        if from_name in self.brain.connectomes:
            if to_area in self.brain.connectomes[from_name]:
                return self.brain.connectomes[from_name][to_area]
        # Try stimulus→area
        if from_name in self.brain.connectomes_by_stimulus:
            if to_area in self.brain.connectomes_by_stimulus[from_name]:
                return self.brain.connectomes_by_stimulus[from_name][to_area]
        return None

    def get_weight_stats(self, from_name: str, to_area: str) -> dict:
        """Get weight statistics for a connectome (area→area or stimulus→area)."""
        try:
            connectome = self._get_connectome(from_name, to_area)
            if connectome is None or connectome.size == 0:
                return {"mean": 0, "std": 0, "min": 0, "max": 0,
                        "nonzero": 0, "mean_nonzero": 0}
            nonzero_vals = connectome[connectome > 0]
            mean_nz = float(np.mean(nonzero_vals)) if len(nonzero_vals) > 0 else 0
            return {
                "mean": float(np.mean(connectome)),
                "mean_nonzero": mean_nz,
                "std": float(np.std(connectome)),
                "min": float(np.min(connectome)),
                "max": float(np.max(connectome)),
                "nonzero": int(np.count_nonzero(connectome)),
                "total": int(connectome.size),
            }
        except (KeyError, IndexError):
            return {"mean": 0, "std": 0, "min": 0, "max": 0,
                    "nonzero": 0, "mean_nonzero": 0}

    def get_weight_sample(self, from_name: str, to_area: str,
                          sample_size: int = 1000) -> list:
        """Get a sample of weight values for histogram (area→area or stim→area)."""
        try:
            connectome = self._get_connectome(from_name, to_area)
            if connectome is None or connectome.size == 0:
                return []
            flat = connectome.flatten()
            if len(flat) > sample_size:
                idx = np.random.choice(len(flat), sample_size, replace=False)
                return flat[idx].tolist()
            return flat.tolist()
        except (KeyError, IndexError):
            return []

    def get_overlap_history(self, area_name: str) -> list[float]:
        """Get overlap-with-previous history for an area across all steps."""
        history = []
        for step in self.steps:
            snap = step.areas_snapshot.get(area_name, {})
            history.append(snap.get("overlap_prev", 0.0))
        return history

    def get_winner_count_history(self, area_name: str) -> list[int]:
        """Get winner count history for an area across all steps."""
        history = []
        for step in self.steps:
            snap = step.areas_snapshot.get(area_name, {})
            history.append(snap.get("n_winners", 0))
        return history

    def get_involvement_history(self, area_name: str) -> list[bool]:
        """Get per-step involvement flags for an area."""
        history = []
        for step in self.steps:
            snap = step.areas_snapshot.get(area_name, {})
            history.append(snap.get("involved", True))
        return history

    # ── DFA Sequence Training & Testing ──────────────────────────────

    def setup_dfa(self, alphabet: str = "ab", n: int = 500,
                  k: int = 50, beta: float = 0.05) -> None:
        """Create standard DFA areas + stimuli.

        Areas: Input, State, Accept, Reject
        Stimuli: one per character in alphabet
        """
        self.reset()
        self.default_beta = beta

        for area_name in ["Input", "State", "Accept", "Reject"]:
            self.add_area(area_name, n=n, k=k, beta=beta)

        for ch in alphabet:
            self.add_stimulus(ch, size=k)

        self._log("setup_dfa",
                  f"DFA setup: alphabet='{alphabet}', areas=Input/State/Accept/Reject",
                  involved_areas={"Input", "State", "Accept", "Reject"})

    def process_string(self, s: str, stabilize: int = 3) -> None:
        """Feed a string character by character through Input→State.

        For each character:
          1. Stimulate char → Input
          2. Project Input → State  (char influences state)
          3. Project State → State × stabilize (let state settle)
        """
        for i, ch in enumerate(s):
            if ch not in self.brain.stimulus_size_by_name:
                continue
            # Feed character into Input
            self.brain.project({ch: ["Input"]}, {})
            # Project Input → State (character drives transition)
            self.brain.project({}, {"Input": ["State"]})
            # Let State stabilize with recurrent connections
            for _ in range(stabilize):
                self.brain.project({}, {"State": ["State"]})

    def train_string(self, s: str, accepted: bool,
                     stabilize: int = 3, reinforce: int = 5) -> None:
        """Process a string, then reinforce Accept or Reject area.

        Args:
            s: the input string
            accepted: True if the string should be accepted
            stabilize: rounds of State→State per character
            reinforce: rounds of State→Accept/Reject at the end
        """
        self.process_string(s, stabilize=stabilize)

        target = "Accept" if accepted else "Reject"
        for _ in range(reinforce):
            self.brain.project({}, {"State": [target]})

        label = "✅ ACCEPT" if accepted else "❌ REJECT"
        self._log("train", f"Train '{s}' → {label}",
                  involved_areas={"Input", "State", target})

    def train_batch(self, accepted: list[str], rejected: list[str],
                    epochs: int = 5, stabilize: int = 3,
                    reinforce: int = 5) -> dict:
        """Train on batches of accepted/rejected strings.

        Returns training summary with epoch count.
        """
        import random
        all_examples = [(s, True) for s in accepted] + \
                       [(s, False) for s in rejected]

        for epoch in range(epochs):
            random.shuffle(all_examples)
            for s, is_accepted in all_examples:
                self.train_string(s, is_accepted,
                                  stabilize=stabilize,
                                  reinforce=reinforce)

        total = len(all_examples) * epochs
        self._log("train_batch",
                  f"Trained {epochs} epochs × {len(all_examples)} strings = {total} total",
                  involved_areas={"Input", "State", "Accept", "Reject"})

        return {
            "epochs": epochs,
            "accepted_count": len(accepted),
            "rejected_count": len(rejected),
            "total_trainings": total,
        }

    def test_string(self, s: str, stabilize: int = 3,
                    probe_rounds: int = 5) -> dict:
        """Test whether a string is accepted or rejected.

        Processes the string, then projects State into both
        Accept and Reject areas and compares assembly strength.

        Returns:
            {string, verdict, accept_score, reject_score, confidence}
        """
        self.process_string(s, stabilize=stabilize)

        # Save State winners before probing
        state_winners = list(self.brain.area_by_name["State"].winners)

        # Probe Accept area
        # Reset Accept/Reject winners so we get clean measurements
        self.brain.area_by_name["Accept"].winners = []
        for _ in range(probe_rounds):
            self.brain.project({}, {"State": ["Accept"]})
        accept_state = self.get_area_state("Accept")
        accept_score = accept_state.get("n_winners", 0)

        # Restore State winners and probe Reject area
        self.brain.area_by_name["State"].winners = state_winners
        self.brain.area_by_name["Reject"].winners = []
        for _ in range(probe_rounds):
            self.brain.project({}, {"State": ["Reject"]})
        reject_state = self.get_area_state("Reject")
        reject_score = reject_state.get("n_winners", 0)

        # Compare by overlap: which area has higher overlap
        # with its previously trained pattern?
        accept_overlap = accept_state.get("overlap_prev", 0)
        reject_overlap = reject_state.get("overlap_prev", 0)

        # Verdict based on overlap (which area recognizes the pattern better)
        if accept_overlap > reject_overlap:
            verdict = "ACCEPT"
        elif reject_overlap > accept_overlap:
            verdict = "REJECT"
        else:
            verdict = "UNCERTAIN"

        total = accept_overlap + reject_overlap
        confidence = abs(accept_overlap - reject_overlap) / total if total > 0 else 0

        result = {
            "string": s,
            "verdict": verdict,
            "accept_overlap": round(accept_overlap, 4),
            "reject_overlap": round(reject_overlap, 4),
            "confidence": round(confidence, 4),
        }

        emoji = "✅" if verdict == "ACCEPT" else "❌" if verdict == "REJECT" else "❓"
        self._log("test",
                  f"Test '{s}' → {emoji} {verdict} "
                  f"(accept={accept_overlap:.3f}, reject={reject_overlap:.3f}, "
                  f"conf={confidence:.2f})",
                  involved_areas={"Input", "State", "Accept", "Reject"})

        return result

    # ── Save / Load ────────────────────────────────────────────────────

    def save_state(self, directory: str) -> str:
        """Save entire brain state to a directory.

        Creates:
          - brain_meta.json  (area/stimulus configs, fibers, learning)
          - brain_weights.npz (all connectome matrices)

        Returns the directory path.
        """
        os.makedirs(directory, exist_ok=True)

        # -- Metadata --
        meta = {
            "p": self._p,
            "seed": self._seed,
            "default_beta": self.default_beta,
            "learning": self.learning,
            "step_counter": self._step_counter,
            "fibers": self._fibers,
            "areas": {},
            "stimuli": {},
        }

        for name, area in self.brain.area_by_name.items():
            meta["areas"][name] = {
                "n": area.n,
                "k": area.k,
                "beta": area.beta,
                "w": area.w,
                "explicit": area.explicit,
                "winners": list(area.winners) if area.winners else [],
                "fixed_assembly": area.fixed_assembly,
                "beta_by_stimulus": area.beta_by_stimulus,
                "beta_by_area": area.beta_by_area,
            }

        for name, size in self.brain.stimulus_size_by_name.items():
            meta["stimuli"][name] = {"size": size}

        meta_path = os.path.join(directory, "brain_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # -- Weight matrices --
        arrays = {}

        # Area→area connectomes
        for from_area, targets in self.brain.connectomes.items():
            for to_area, matrix in targets.items():
                key = f"conn__{from_area}__{to_area}"
                arrays[key] = matrix

        # Stimulus→area connectomes
        for stim, targets in self.brain.connectomes_by_stimulus.items():
            for to_area, matrix in targets.items():
                key = f"stim__{stim}__{to_area}"
                arrays[key] = matrix

        weights_path = os.path.join(directory, "brain_weights.npz")
        np.savez_compressed(weights_path, **arrays)

        return directory

    def load_state(self, directory: str) -> None:
        """Load brain state from a directory created by save_state().

        Completely replaces the current brain and LiveBrain state.
        """
        meta_path = os.path.join(directory, "brain_meta.json")
        weights_path = os.path.join(directory, "brain_weights.npz")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        weights = np.load(weights_path)

        # Rebuild brain from scratch
        p = meta["p"]
        seed = meta.get("seed", 0)
        self._p = p
        self._seed = seed
        self.default_beta = meta["default_beta"]
        self.learning = meta["learning"]
        self._step_counter = meta.get("step_counter", 0)
        self._fibers = [tuple(f) for f in meta.get("fibers", [])]

        self.brain = Brain(p, save_size=True, save_winners=True, seed=seed)

        # Recreate areas (using explicit to get full connectomes)
        for name, cfg in meta["areas"].items():
            area = Area(
                name, cfg["n"], cfg["k"],
                beta=cfg["beta"], w=cfg["w"],
                explicit=cfg.get("explicit", True),
            )
            area.winners = cfg.get("winners", [])
            area.fixed_assembly = cfg.get("fixed_assembly", False)
            area.beta_by_stimulus = cfg.get("beta_by_stimulus", {})
            area.beta_by_area = cfg.get("beta_by_area", {})
            if hasattr(area, 'ever_fired'):
                area.ever_fired = np.zeros(cfg["n"], dtype=bool)
                if area.winners:
                    for w in area.winners:
                        if w < cfg["n"]:
                            area.ever_fired[w] = True
                area.num_ever_fired = int(np.sum(area.ever_fired))
            self.brain.area_by_name[name] = area
            self._prev_winners[name] = set(area.winners) if area.winners else set()

        # Recreate stimuli
        for name, cfg in meta["stimuli"].items():
            self.brain.stimulus_size_by_name[name] = cfg["size"]
            self.brain.connectomes_by_stimulus[name] = {}

        # Initialize empty connectome dicts for each area
        for area_name in self.brain.area_by_name:
            self.brain.connectomes[area_name] = {}

        # Load weight matrices
        for key in weights.files:
            parts = key.split("__")
            if parts[0] == "conn" and len(parts) == 3:
                from_area, to_area = parts[1], parts[2]
                if from_area not in self.brain.connectomes:
                    self.brain.connectomes[from_area] = {}
                self.brain.connectomes[from_area][to_area] = weights[key]
            elif parts[0] == "stim" and len(parts) == 3:
                stim, to_area = parts[1], parts[2]
                if stim not in self.brain.connectomes_by_stimulus:
                    self.brain.connectomes_by_stimulus[stim] = {}
                self.brain.connectomes_by_stimulus[stim][to_area] = weights[key]

        weights.close()

        # Clear step history (it's a new session)
        self.steps.clear()
        self._last_snapshot.clear()
        self._log("load", f"Loaded brain from '{directory}'",
                  involved_areas=set(self.brain.area_by_name.keys()))

    # ── Internal ────────────────────────────────────────────────────────

    def _log(self, operation: str, description: str,
             involved_areas: set = None) -> StepRecord:
        """Log current state as a step.

        Args:
            involved_areas: set of area names that were actually projected to
                in this operation. Only these areas get their prev_winners
                updated. Uninvolved areas keep their last real overlap and
                are marked with involved=False in the snapshot.
        """
        if involved_areas is None:
            involved_areas = set(self.brain.area_by_name.keys())

        snapshot = {}
        for name in self.brain.area_by_name:
            state = self.get_area_state(name)
            if name in involved_areas:
                state["involved"] = True
            else:
                # Not involved: carry forward last real overlap value
                prev_snap = self._last_snapshot.get(name, {})
                state["overlap_prev"] = prev_snap.get("overlap_prev", 0.0)
                state["gained"] = 0
                state["lost"] = 0
                state["gained_set"] = set()
                state["lost_set"] = set()
                state["involved"] = False
            snapshot[name] = state

        # Cache snapshot for display (before updating prev_winners!)
        self._last_snapshot = snapshot

        # Only update prev_winners for areas that were actually involved
        for name in involved_areas:
            if name in self.brain.area_by_name:
                area = self.brain.area_by_name[name]
                self._prev_winners[name] = set(area.winners) if area.winners else set()

        record = StepRecord(
            step=self._step_counter,
            operation=operation,
            description=description,
            areas_snapshot=snapshot,
            fibers=list(self._fibers),
            learning=self.learning,
        )
        self.steps.append(record)
        self._step_counter += 1
        return record
