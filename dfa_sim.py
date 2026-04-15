"""
DFA (Deterministic Finite Automaton) simulation using the Assembly Calculus model.

Architecture:
    STATE  (explicit) -- represents *all* DFA states as pre-formed assemblies.
    INPUT  (explicit) -- represents *all* alphabet symbols as pre-formed assemblies.
    EDGES  (explicit) -- one pre-formed assembly per (state, symbol) pair.
                         During training, we learn the mapping from EDGES -> RESULT.
    RESULT (explicit) -- same layout as STATE; readout area for the next state.

Training (for each delta(qi, sj) = qk):
    Phase 1: Activate STATE=qi, INPUT=sj, fix EDGES(qi,sj).
             Project {STATE, INPUT} -> EDGES repeatedly.
             Hebbian plasticity strengthens STATE(qi)->EDGES(ci) and INPUT(sj)->EDGES(ci).
    Phase 2: Activate EDGES(qi,sj), fix RESULT=qk.
             Project EDGES -> RESULT repeatedly.
             Hebbian plasticity strengthens EDGES(ci)->RESULT(qk).

Runtime:
    1. Activate STATE=qi (fixed), INPUT=sj (fixed).
    2. Project {STATE, INPUT} -> EDGES  (the combined signal selects the
       correct pre-formed (qi,sj) assembly in EDGES).
    3. Unfix RESULT.
    4. Project EDGES -> RESULT  (learned connections select qk).
    5. Readout RESULT.

Key insight: Using an explicit EDGES area with pre-formed assemblies
guarantees that each (state, input) pair has a distinct, non-overlapping
assembly. This eliminates the interference problem that arises with
non-explicit areas where strong shared connections can collapse
multiple assemblies into one.
"""

import brain
import brain_util as bu
import numpy as np

# Compatibility patch used elsewhere in this codebase
brain.Brain.areas = property(lambda self: self.area_by_name)


class DFABrain(brain.Brain):
    """Assembly-calculus implementation of a Deterministic Finite Automaton."""

    def __init__(
        self,
        states,
        alphabet,
        start_state,
        accept_states,
        transitions,
        *,
        p=0.05,
        k=100,
        beta=0.05,
        seed=0,
    ):
        """
        Args:
            states:        list of state names, e.g. ["q0", "q1", "q_trash"]
            alphabet:      list of symbol names, e.g. ["a", "b"]
            start_state:   name of the initial state
            accept_states: iterable of accepting-state names
            transitions:   dict  (state, symbol) -> next_state
                           Must be total (every state x symbol pair present).
            p:             neuron connection probability
            k:             assembly size (neurons per winner set)
            beta:          Hebbian plasticity rate
            seed:          RNG seed for reproducibility
        """
        super().__init__(p, save_winners=False, seed=seed)

        self.states = list(states)
        self.alphabet = list(alphabet)
        self.start_state = start_state
        self.accept_states = set(accept_states)
        self.transitions = dict(transitions)
        self.k_val = k

        # index lookups
        self.state_index = {s: i for i, s in enumerate(self.states)}
        self.symbol_index = {s: i for i, s in enumerate(self.alphabet)}

        num_states = len(self.states)
        num_symbols = len(self.alphabet)
        num_edges = num_states * num_symbols

        # --- brain areas ---
        self.add_explicit_area("STATE", num_states * k, k, beta)
        self.add_explicit_area("INPUT", num_symbols * k, k, beta)
        # EDGES: one assembly per (state, symbol) pair — guarantees separation.
        self.add_explicit_area("EDGES", num_edges * k, k, beta)
        self.add_explicit_area("RESULT", num_states * k, k, beta)

        self.current_state = start_state
        self._trained = False

        # Training instrumentation
        self.training_log = []  # list of snapshot dicts

    def _edges_index(self, state_name, symbol_name):
        """Map (state, symbol) to a unique EDGES assembly index."""
        si = self.state_index[state_name]
        ai = self.symbol_index[symbol_name]
        return si * len(self.alphabet) + ai

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, rounds=30, record=True):
        """Train every transition via Hebbian plasticity.

        Phase 1: Wire STATE + INPUT -> EDGES  (learn which edges assembly
                 responds to which (state, input) pair).
        Phase 2: Wire EDGES -> RESULT  (learn the transition output).

        Args:
            rounds: number of Hebbian projection rounds per association.
            record: if True, record training snapshots for visualization.
        """
        self.training_log = []

        # Phase 1: for each (state, symbol) pair, strengthen connections
        # from STATE(qi) + INPUT(sj) to EDGES(qi,sj).
        for state_name in self.states:
            for symbol_name in self.alphabet:
                self._train_edges_association(state_name, symbol_name,
                                              rounds, record)

        # Phase 2: for each transition, strengthen EDGES -> RESULT.
        for (src, sym), dst in self.transitions.items():
            self._train_transition(src, sym, dst, rounds, record)

        self._trained = True

    def _train_edges_association(self, state_name, symbol_name, rounds, record):
        """Train: when STATE=qi and INPUT=sj fire, EDGES(qi,sj) should win."""
        si = self.state_index[state_name]
        ai = self.symbol_index[symbol_name]
        ci = self._edges_index(state_name, symbol_name)

        for r in range(rounds):
            self.activate("STATE", si)
            self.activate("INPUT", ai)
            self.activate("EDGES", ci)
            # Project both into EDGES (EDGES is fixed -> plasticity strengthens
            # STATE(qi)->EDGES(ci) and INPUT(sj)->EDGES(ci) connections).
            self.project({}, {"STATE": ["EDGES"], "INPUT": ["EDGES"]})

            if record:
                self._record_snapshot(
                    phase="phase1",
                    transition=f"({state_name},{symbol_name})",
                    round_num=r + 1,
                    total_rounds=rounds,
                )

    def _train_transition(self, src, sym, dst, rounds, record):
        """Train: EDGES(src, sym) -> RESULT(dst)."""
        ci = self._edges_index(src, sym)
        dst_idx = self.state_index[dst]

        for r in range(rounds):
            self.activate("EDGES", ci)
            self.activate("RESULT", dst_idx)
            # EDGES is fixed as source, RESULT is fixed as target.
            # Plasticity strengthens EDGES(ci) -> RESULT(dst_idx).
            self.project({}, {"EDGES": ["RESULT"]})

            if record:
                self._record_snapshot(
                    phase="phase2",
                    transition=f"({src},{sym})->{dst}",
                    round_num=r + 1,
                    total_rounds=rounds,
                )

    def _record_snapshot(self, phase, transition, round_num, total_rounds):
        """Record a lightweight training snapshot for visualization."""
        # Get weight stats for key connectomes
        def _stats(from_area, to_area):
            conn = self.connectomes.get(from_area, {}).get(to_area)
            if conn is None or conn.size == 0:
                return {"mean": 0, "max": 0, "std": 0}
            nz = conn[conn > 0]
            if len(nz) == 0:
                return {"mean": 0, "max": 0, "std": 0}
            return {
                "mean": float(np.mean(nz)),
                "max": float(np.max(nz)),
                "std": float(np.std(nz)),
            }

        snap = {
            "step": len(self.training_log),
            "phase": phase,
            "transition": transition,
            "round": round_num,
            "total_rounds": total_rounds,
            "weights": {
                "STATE->EDGES": _stats("STATE", "EDGES"),
                "INPUT->EDGES": _stats("INPUT", "EDGES"),
                "EDGES->RESULT": _stats("EDGES", "RESULT"),
            },
        }

        # During phase 2, also record readout confidence for this transition
        if phase == "phase2" and round_num == total_rounds:
            snap["readout"] = self._test_all_transitions()

        self.training_log.append(snap)

    def _test_all_transitions(self):
        """Quick readout test of all transitions. Returns dict of confidence."""
        results = {}
        self.disable_plasticity = True
        for (src, sym), expected_dst in self.transitions.items():
            self.activate("STATE", self.state_index[src])
            self.activate("INPUT", self.symbol_index[sym])
            self.area_by_name["EDGES"].unfix_assembly()
            self.project({}, {"STATE": ["EDGES"], "INPUT": ["EDGES"]})
            self.area_by_name["RESULT"].unfix_assembly()
            self.project({}, {"EDGES": ["RESULT"]})
            actual, confidence = self._readout_state()
            results[f"({src},{sym})->{expected_dst}"] = {
                "expected": expected_dst,
                "actual": actual,
                "confidence": confidence,
                "correct": actual == expected_dst,
            }
        self.disable_plasticity = False
        return results

    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------
    def step(self, symbol):
        """Process one input symbol.  Returns (new_state, confidence)."""
        if not self._trained:
            raise RuntimeError("Call train() before step()")

        self.disable_plasticity = True

        if symbol not in self.symbol_index:
            self.disable_plasticity = False
            raise ValueError(f"Unknown symbol: {symbol!r}")

        # 1. Activate current state and input symbol (both fixed).
        self.activate("STATE", self.state_index[self.current_state])
        self.activate("INPUT", self.symbol_index[symbol])

        # 2. Project STATE + INPUT -> EDGES (unfixed, to let the brain pick).
        self.area_by_name["EDGES"].unfix_assembly()
        self.project({}, {"STATE": ["EDGES"], "INPUT": ["EDGES"]})

        # 3. Project EDGES -> RESULT (unfixed).
        self.area_by_name["RESULT"].unfix_assembly()
        self.project({}, {"EDGES": ["RESULT"]})

        # 4. Readout.
        new_state, confidence = self._readout_state()
        self.current_state = new_state

        self.disable_plasticity = False
        return new_state, confidence

    def run(self, input_string):
        """Run the DFA on a full input string.

        Returns:
            (accepted: bool, final_state: str, trace: list[str])
        """
        self.reset()
        trace = [self.current_state]
        for sym in input_string:
            if sym not in self.symbol_index:
                raise ValueError(f"Unknown symbol: {sym!r}")
            new_state, _ = self.step(sym)
            trace.append(new_state)
        accepted = self.current_state in self.accept_states
        return accepted, self.current_state, trace

    def run_with_snapshots(self, input_string):
        """Run the DFA on a string, capturing full neuron snapshots at each step.

        Returns a list of snapshot dicts, one per step (step 0 = initial state).
        Each snapshot contains:
            step:       int (0-based)
            symbol:     str or None (None for step 0)
            state_from: str (state before this symbol)
            state_to:   str (state after this symbol, = state_from for step 0)
            accepted:   bool (is state_to an accept state?)
            confidence: float (readout confidence, 0 for step 0)
            areas:      dict mapping area name -> {
                            winners: list[int],
                            assembly_label: str,
                            assembly_index: int,
                        }
        """
        self.reset()
        snapshots = []
        k = self.k_val

        def _identify_assembly(area_name, winners_set):
            """Identify which pre-formed assembly best matches the winners."""
            if area_name == "STATE" or area_name == "RESULT":
                labels = self.states
            elif area_name == "INPUT":
                labels = self.alphabet
            elif area_name == "EDGES":
                labels = []
                for s in self.states:
                    for sym in self.alphabet:
                        labels.append(f"{s},{sym}")
            else:
                return "?", -1, 0.0

            best_label = "?"
            best_idx = -1
            best_overlap = 0
            for idx, label in enumerate(labels):
                start = idx * k
                asm = set(range(start, start + k))
                ov = len(winners_set & asm)
                if ov > best_overlap:
                    best_overlap = ov
                    best_label = label
                    best_idx = idx
            pct = best_overlap / k if k > 0 else 0.0
            return best_label, best_idx, pct

        def _capture(step_num, symbol, state_from, state_to, confidence):
            areas = {}
            for area_name in ["STATE", "INPUT", "EDGES", "RESULT"]:
                area = self.area_by_name[area_name]
                winners = list(area.winners) if area.winners else []
                winners_set = set(winners)
                label, idx, pct = _identify_assembly(area_name, winners_set)
                areas[area_name] = {
                    "winners": winners,
                    "assembly_label": label,
                    "assembly_index": idx,
                    "assembly_pct": round(pct, 3),
                }
            return {
                "step": step_num,
                "symbol": symbol,
                "state_from": state_from,
                "state_to": state_to,
                "accepted": state_to in self.accept_states,
                "confidence": confidence,
                "areas": areas,
            }

        # Step 0: initial state (just activate STATE, no input)
        self.disable_plasticity = True
        self.activate("STATE", self.state_index[self.current_state])
        snapshots.append(_capture(0, None, self.current_state,
                                  self.current_state, 1.0))

        # Steps 1..n: process each character
        for i, sym in enumerate(input_string):
            state_from = self.current_state
            new_state, conf = self.step(sym)
            snapshots.append(_capture(i + 1, sym, state_from,
                                      new_state, conf))

        self.disable_plasticity = False
        return snapshots

    def reset(self):
        """Reset to start state."""
        self.current_state = self.start_state

    # ------------------------------------------------------------------
    # Readout
    # ------------------------------------------------------------------
    def _readout_state(self):
        """Identify which state assembly is active in RESULT.

        Returns (state_name, confidence) where confidence = overlap / k.
        """
        winners = set(self.area_by_name["RESULT"].winners)
        k = self.k_val
        best_overlap = -1
        best_state = None
        for name, idx in self.state_index.items():
            start = idx * k
            assembly = set(range(start, start + k))
            ov = len(winners & assembly)
            if ov > best_overlap:
                best_overlap = ov
                best_state = name
        confidence = best_overlap / k if k > 0 else 0.0
        return best_state, confidence


# ======================================================================
# Helper: build DFA from regex
# ======================================================================
def build_from_regex(pattern, alphabet, **kwargs):
    """Build a DFABrain from a regex pattern.

    Args:
        pattern:  regex string, e.g. "(ab)*"
        alphabet: list of characters, e.g. ["a", "b"]
        **kwargs: passed to DFABrain (p, k, beta, seed)

    Returns:
        DFABrain instance (untrained).
    """
    from regex_to_dfa import regex_to_dfa
    states, alpha, start, accept, trans = regex_to_dfa(pattern, alphabet)
    return DFABrain(
        states=states,
        alphabet=alpha,
        start_state=start,
        accept_states=accept,
        transitions=trans,
        **kwargs,
    )


# ======================================================================
# Helper: build the (ab)* DFA
# ======================================================================
def build_ab_star_dfa(**kwargs):
    """Return a DFABrain for the language (ab)*.

    States:  q0 (start, accept), q1, q_trash
    Alphabet: a, b
    Transitions:
        q0 --a--> q1
        q0 --b--> q_trash
        q1 --a--> q_trash
        q1 --b--> q0
        q_trash --a--> q_trash
        q_trash --b--> q_trash
    """
    states = ["q0", "q1", "q_trash"]
    alphabet = ["a", "b"]
    transitions = {
        ("q0", "a"): "q1",
        ("q0", "b"): "q_trash",
        ("q1", "a"): "q_trash",
        ("q1", "b"): "q0",
        ("q_trash", "a"): "q_trash",
        ("q_trash", "b"): "q_trash",
    }
    return DFABrain(
        states=states,
        alphabet=alphabet,
        start_state="q0",
        accept_states=["q0"],
        transitions=transitions,
        **kwargs,
    )


# ======================================================================
# Helper: build the (aba)*c*ba DFA
# ======================================================================
def build_abacba_dfa(**kwargs):
    """Return a DFABrain for the language (aba)*c*ba.

    Accepts strings of the form:  (aba)^n  c^m  ba   where n>=0, m>=0.
    Alphabet: {a, b, c}

    States:
        q0  -- start / just completed an "aba" block
        q1  -- seen 'a' (first char of "aba")
        q2  -- seen 'ab' (inside "aba")
        q3  -- in c* section
        q4  -- seen 'b' (first char of final "ba")
        q5  -- accept: completed "ba"
        q_trash -- reject sink

    Transitions:
        q0 --a--> q1    q0 --b--> q4    q0 --c--> q3
        q1 --a--> trash q1 --b--> q2    q1 --c--> trash
        q2 --a--> q0    q2 --b--> trash q2 --c--> trash
        q3 --a--> trash q3 --b--> q4    q3 --c--> q3
        q4 --a--> q5    q4 --b--> trash q4 --c--> trash
        q5 --a--> trash q5 --b--> trash q5 --c--> trash
        trash --*--> trash
    """
    states = ["q0", "q1", "q2", "q3", "q4", "q5", "q_trash"]
    alphabet = ["a", "b", "c"]
    T = "q_trash"
    transitions = {
        ("q0", "a"): "q1",   ("q0", "b"): "q4",   ("q0", "c"): "q3",
        ("q1", "a"): T,      ("q1", "b"): "q2",   ("q1", "c"): T,
        ("q2", "a"): "q0",   ("q2", "b"): T,      ("q2", "c"): T,
        ("q3", "a"): T,      ("q3", "b"): "q4",   ("q3", "c"): "q3",
        ("q4", "a"): "q5",   ("q4", "b"): T,      ("q4", "c"): T,
        ("q5", "a"): T,      ("q5", "b"): T,      ("q5", "c"): T,
        (T, "a"): T,         (T, "b"): T,         (T, "c"): T,
    }
    return DFABrain(
        states=states,
        alphabet=alphabet,
        start_state="q0",
        accept_states=["q5"],
        transitions=transitions,
        **kwargs,
    )


# ======================================================================
# Quick demo
# ======================================================================
def _run_demo(name, dfa_builder, test_cases, **build_kwargs):
    """Run a DFA demo with the given test cases."""
    print(f"{'=' * 60}")
    print(f"  DFA: {name}")
    print(f"{'=' * 60}")
    dfa = dfa_builder(**build_kwargs)
    print("Training ...")
    dfa.train(rounds=30, record=False)
    print("Training complete.\n")

    passed = 0
    for input_str, expected in test_cases:
        accepted, final, trace = dfa.run(input_str)
        status = "PASS" if accepted == expected else "FAIL"
        if status == "PASS":
            passed += 1
        display = input_str if input_str else "<empty>"
        print(f"  [{status}]  input={display!s:12s}  accepted={accepted!s:5s}  "
              f"expected={expected!s:5s}  trace={' -> '.join(trace)}")

    print(f"\n  {passed}/{len(test_cases)} tests passed.\n")
    return passed == len(test_cases)


if __name__ == "__main__":
    all_ok = True

    # --- (ab)* ---
    all_ok &= _run_demo(
        "(ab)*", build_ab_star_dfa,
        [
            ("", True),
            ("ab", True),
            ("abab", True),
            ("ababab", True),
            ("a", False),
            ("b", False),
            ("ba", False),
            ("aa", False),
            ("bb", False),
            ("aba", False),
            ("abba", False),
            ("ababb", False),
        ],
        k=100, beta=0.05, p=0.05, seed=42,
    )

    # --- (aba)*c*ba ---
    all_ok &= _run_demo(
        "(aba)*c*ba", build_abacba_dfa,
        [
            # Accept
            ("ba", True),           # n=0 m=0
            ("cba", True),          # n=0 m=1
            ("ccba", True),         # n=0 m=2
            ("ababa", True),        # n=1 m=0: aba + ba
            ("abacba", True),       # n=1 m=1: aba + c + ba
            ("abaccba", True),      # n=1 m=2: aba + cc + ba
            ("abaababa", True),     # n=2 m=0: aba + aba + ba
            ("abaabacba", True),    # n=2 m=1: aba + aba + c + ba
            # Reject
            ("", False),            # empty
            ("a", False),
            ("b", False),
            ("c", False),
            ("ab", False),
            ("aba", False),         # no final "ba"
            ("abc", False),
            ("bba", False),
            ("abba", False),
            ("abaca", False),
            ("cca", False),
            ("baab", False),
        ],
        k=100, beta=0.05, p=0.05, seed=42,
    )

    # --- regex-built DFA ---
    print("=" * 60)
    print("  DFA from regex: (ab)*")
    print("=" * 60)
    dfa = build_from_regex("(ab)*", ["a", "b"], k=100, beta=0.05, p=0.05, seed=42)
    dfa.train(rounds=30, record=False)
    for s, exp in [("", True), ("ab", True), ("abab", True), ("a", False), ("ba", False)]:
        acc, _, trace = dfa.run(s)
        status = "PASS" if acc == exp else "FAIL"
        print(f"  [{status}]  '{s or '<empty>'}' -> {acc}")

    print("\nALL PASSED!" if all_ok else "SOME TESTS FAILED.")
