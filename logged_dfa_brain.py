"""
LoggedDFABrain — DFABrain subclass that instruments run_inference()
to emit per-step SimulationLogger events with full attribution.
"""

import numpy as np
from collections import defaultdict
from dfa_brain import DFABrain
from simulation_logger import SimulationLogger


class LoggedDFABrain(DFABrain):
    """DFABrain with automatic per-step logging via SimulationLogger."""

    def __init__(self, p, n=1000, k=100, beta=0.05, log_dir="logs"):
        super().__init__(p, n=n, k=k, beta=beta)
        self.logger = SimulationLogger(log_dir=log_dir)

    def finalize_machine(self):
        """Finalize and auto-register templates for all states + transitions."""
        super().finalize_machine()
        # Register state templates
        for s in self.states:
            self.logger.register_template(
                name=s, area=s, neuron_ids=list(range(self.k))
            )
        # Register transition sub-assembly templates
        for i, ((q, sym), q_next) in enumerate(self.transitions_list):
            offset = i * self.k
            self.logger.register_template(
                name=f"trans_{q}_{sym}_{q_next}",
                area=self.trans_area_name,
                neuron_ids=list(range(offset, offset + self.k)),
            )

    def run_inference(self, input_str):
        """Runs the DFA on input string with full per-step logging."""
        history = []
        current_state = "q0"

        # Reset Areas
        for name, area in self.area_by_name.items():
            area.winners = []
            if name == self.trans_area_name:
                area.w = self.total_t_neurons
            else:
                area.w = self.k

        self.activate(current_state, 0)
        history.append(current_state)

        # Log initial state
        init_activations = {}
        for s in self.states:
            s_area = self.area_by_name[s]
            init_activations[s] = self.logger.record_activation(
                area=s,
                winners=s_area.winners,
                winner_inputs=[1.0] * len(s_area.winners) if s_area.winners else [],
            )
        self.logger.log_step(
            activations=init_activations,
            phase="init",
            dfa_state=current_state,
        )

        for i, char in enumerate(input_str):
            # --- Phase 1: Activate Transition ---
            t_area = self.area_by_name[self.trans_area_name]
            inputs = np.zeros(t_area.n, dtype=np.float32)

            # Input from State — track attribution
            input_from_state = 0.0
            if (current_state in self.connectomes
                    and self.trans_area_name in self.connectomes[current_state]):
                W_state = self.connectomes[current_state][self.trans_area_name]
                s_winners = self.area_by_name[current_state].winners
                for w_idx in s_winners:
                    inputs += W_state[w_idx]
                input_from_state = float(np.sum(inputs))

            # Input from Symbol
            input_before_sym = float(np.sum(inputs))
            if (char in self.connectomes_by_stimulus
                    and self.trans_area_name in self.connectomes_by_stimulus[char]):
                W_sym = self.connectomes_by_stimulus[char][self.trans_area_name]
                inputs += W_sym
            input_from_symbol = float(np.sum(inputs)) - input_before_sym

            # Winner Take All (k-WTA)
            if self.k < len(inputs):
                potential_winners = np.argpartition(inputs, -self.k)[-self.k:]
            else:
                potential_winners = np.arange(len(inputs))

            t_area.winners = potential_winners.tolist()
            winner_inputs_t = inputs[potential_winners].tolist()

            # Decode Transition
            votes = defaultdict(int)
            for w in t_area.winners:
                votes[w // self.k] += 1
            max_votes = 0
            best_trans_idx = -1
            for idx, count in votes.items():
                if count > max_votes:
                    max_votes = count
                    best_trans_idx = idx

            if best_trans_idx < len(self.transitions_list) and max_votes > 0:
                (q_prev, sym_prev), q_target = self.transitions_list[best_trans_idx]
                transition_str = f"({q_prev}, {sym_prev}) → {q_target}"
            else:
                # Log the failure
                trans_act = self.logger.record_activation(
                    area=self.trans_area_name,
                    winners=t_area.winners,
                    winner_inputs=winner_inputs_t,
                    all_inputs=inputs,
                    attribution={f"state_{current_state}": input_from_state,
                                 f"stim_{char}": input_from_symbol},
                )
                self.logger.log_step(
                    activations={self.trans_area_name: trans_act},
                    phase="project",
                    dfa_state=current_state,
                    dfa_symbol=char,
                    dfa_transition="FAILED",
                )
                return False, history

            # --- Phase 2: Activate Next State ---
            self.area_by_name[current_state].winners = []

            state_scores = {}
            state_inputs_map = {}
            for s in self.states:
                s_area = self.area_by_name[s]
                s_inputs = np.zeros(s_area.n, dtype=np.float32)

                if (self.trans_area_name in self.connectomes
                        and s in self.connectomes[self.trans_area_name]):
                    W_trans = self.connectomes[self.trans_area_name][s]
                    for w_idx in t_area.winners:
                        s_inputs += W_trans[w_idx]

                if self.k < len(s_inputs):
                    s_winners = np.argpartition(s_inputs, -self.k)[-self.k:]
                else:
                    s_winners = np.arange(len(s_inputs))

                canonical_strength = float(np.sum(s_inputs[0:self.k]))
                state_scores[s] = canonical_strength
                state_inputs_map[s] = (s_inputs, s_winners)
                if canonical_strength > 0:
                    s_area.winners = s_winners.tolist()
                else:
                    s_area.winners = []

            best_s = None
            best_score = 0
            for s, score in state_scores.items():
                if score > best_score:
                    best_score = score
                    best_s = s

            # --- Log everything for this step ---
            step_activations = {}

            # Transition area activation
            step_activations[self.trans_area_name] = self.logger.record_activation(
                area=self.trans_area_name,
                winners=t_area.winners,
                winner_inputs=winner_inputs_t,
                all_inputs=inputs,
                attribution={
                    f"state_{current_state}": input_from_state,
                    f"stim_{char}": input_from_symbol,
                },
            )

            # State area activations
            for s in self.states:
                s_area = self.area_by_name[s]
                s_inputs_arr, s_winners = state_inputs_map[s]
                step_activations[s] = self.logger.record_activation(
                    area=s,
                    winners=s_area.winners,
                    winner_inputs=s_inputs_arr[s_winners].tolist() if len(s_area.winners) > 0 else [],
                    all_inputs=s_inputs_arr,
                    attribution={self.trans_area_name: float(np.sum(s_inputs_arr))},
                )

            if best_s:
                history.append(best_s)
                current_state = best_s
                self.activate(current_state, 0)

                self.logger.log_step(
                    activations=step_activations,
                    phase="project",
                    dfa_state=current_state,
                    dfa_symbol=char,
                    dfa_transition=transition_str,
                )
            else:
                self.logger.log_step(
                    activations=step_activations,
                    phase="project",
                    dfa_state=current_state,
                    dfa_symbol=char,
                    dfa_transition="FAILED",
                )
                return False, history

        is_accept = current_state in self.accept_states
        return is_accept, history

    def save_log(self):
        """Write all logged steps to disk."""
        self.logger.save()
        return self.logger.log_path
