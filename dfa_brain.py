import brain
import numpy as np
from collections import defaultdict

class DFABrain(brain.Brain):
    """
    A Brain implementation that acts as a Deterministic Finite Automaton.
    States are represented by Brain Areas.
    Alphabet symbols are represented by Stimuli.
    Transitions are implemented by a single 'Transitions' interneuron area with Global Winner-Take-All competition.
    """
    def __init__(self, p, n=1000, k=100, beta=0.05):
        super().__init__(p, save_size=True, save_winners=True)
        self.n = n
        self.k = k
        self.beta = beta
        self.states = []
        self.alphabet = []
        self.transitions = {} # (state, symbol) -> next_state
        self.start_state = None
        self.accept_states = []
        pass

    def add_state(self, name, is_start=False, is_accept=False):
        self.add_area(name, self.n, self.k, self.beta)
        self.states.append(name)
        if is_start:
            self.start_state = name
        if is_accept:
            self.accept_states.append(name)

    def add_symbol(self, symbol):
        # Symbol is a stimulus
        if symbol not in self.alphabet:
            self.add_stimulus(symbol, self.k) # Stimulus size = k
            self.alphabet.append(symbol)
            pass

    def set_transition(self, from_state, symbol, to_state):
        self.transitions[(from_state, symbol)] = to_state
        
    def finalize_machine(self):
        """
        Builds the DFA circuitry using a SINGLE Transitions area.
        Each transition (q, s) -> q_next corresponds to a generic sub-assembly in the Transitions area.
        We wire (q + s) -> Transition_SubAssembly -> q_next.
        Global k-WTA in the Transitions area ensures only the strongest transition fires.
        """
        print("Finalizing DFA with Single Transition Area...")
        
        # 1. Assign unique offsets for each transition to ensure separation
        self.transitions_list = list(self.transitions.items()) 
        # Sort for determinism
        self.transitions_list.sort(key=lambda x: str(x[0]))
            
        self.num_transitions = len(self.transitions_list)
        self.trans_area_name = "Transitions"
        self.total_t_neurons = self.num_transitions * self.k
        
        # Create Transitions area
        self.add_area(self.trans_area_name, self.total_t_neurons, self.k, beta=0)
        
        # Unfix and set size to total_n to allow manual management of all neurons
        t_area = self.area_by_name[self.trans_area_name]
        t_area.w = self.total_t_neurons
        t_area.unfix_assembly() 
        
        # Initialize state assemblies
        for s in self.states:
            self.activate(s, 0)
            self.area_by_name[s].w = self.k
            self.area_by_name[s].unfix_assembly()
            
        STRONG_WEIGHT = 5.0
        
        def wire_subregion(source, target_area, target_offset, weight):
            t_area_obj = self.area_by_name[target_area]
            
            if hasattr(self, 'connectomes_by_stimulus') and source in self.connectomes_by_stimulus:
                c = self.connectomes_by_stimulus[source][target_area]
                if c.shape != (t_area_obj.n,):
                    # Only create fresh zeroes if shape mismatches (first time seeing this target)
                    c = np.zeros(t_area_obj.n, dtype=np.float32)
                    self.connectomes_by_stimulus[source][target_area] = c
                
                # Accumulate/Set weights
                c[target_offset : target_offset + self.k] = weight
            else:
                if source not in self.connectomes: self.connectomes[source] = {}
                if target_area not in self.connectomes[source]:
                     self.connectomes[source][target_area] = np.zeros(
                         (self.area_by_name[source].n, t_area_obj.n), dtype=np.float32)
                
                c = self.connectomes[source][target_area]
                if c.shape != (self.area_by_name[source].n, t_area_obj.n):
                     new_c = np.zeros((self.area_by_name[source].n, t_area_obj.n), dtype=np.float32)
                     r = min(c.shape[0], new_c.shape[0])
                     co = min(c.shape[1], new_c.shape[1])
                     new_c[:r, :co] = c[:r, :co]
                     c = new_c
                     self.connectomes[source][target_area] = c

                c[0 : self.k, target_offset : target_offset + self.k] = weight

        def wire_out_subregion(source_area, source_offset, target, weight):
             if source_area not in self.connectomes: self.connectomes[source_area] = {}
             target_area_obj = self.area_by_name[target]
             if target not in self.connectomes[source_area]:
                 self.connectomes[source_area][target] = np.zeros(
                     (self.area_by_name[source_area].n, target_area_obj.n), dtype=np.float32)
                     
             c = self.connectomes[source_area][target]
             if c.shape != (self.area_by_name[source_area].n, target_area_obj.n):
                 new_c = np.zeros((self.area_by_name[source_area].n, target_area_obj.n), dtype=np.float32)
                 r = min(c.shape[0], new_c.shape[0])
                 co = min(c.shape[1], new_c.shape[1])
                 new_c[:r, :co] = c[:r, :co]
                 c = new_c
                 self.connectomes[source_area][target] = c
                 
             c[source_offset : source_offset + self.k, 0 : self.k] = weight

        # Wire transitions
        for i, ((q, sym), q_next) in enumerate(self.transitions_list):
            offset = i * self.k
            # Input: State + Symbol -> Transition Sub-region
            wire_subregion(q, self.trans_area_name, offset, STRONG_WEIGHT)
            wire_subregion(sym, self.trans_area_name, offset, STRONG_WEIGHT)
            # Output: Transition Sub-region -> Next State
            wire_out_subregion(self.trans_area_name, offset, q_next, STRONG_WEIGHT)
            
        self.disable_plasticity = True
        
        # Robust Initialization of all potential connectomes to avoid broadcasting errors
        all_area_names = list(self.area_by_name.keys())
        for target_name in all_area_names:
            target_area = self.area_by_name[target_name]
            for src in all_area_names:
                if src not in self.connectomes: self.connectomes[src] = {}
                expected_shape = (self.area_by_name[src].n, target_area.n)
                if target_name not in self.connectomes[src]:
                    self.connectomes[src][target_name] = np.zeros(expected_shape, dtype=np.float32)
                else:
                    c = self.connectomes[src][target_name]
                    if c.shape != expected_shape:
                        new_c = np.zeros(expected_shape, dtype=np.float32)
                        r = min(c.shape[0], new_c.shape[0])
                        co = min(c.shape[1], new_c.shape[1])
                        new_c[:r, :co] = c[:r, :co]
                        self.connectomes[src][target_name] = new_c
                        
            for sym in self.alphabet:
                 if hasattr(self, 'connectomes_by_stimulus') and sym in self.connectomes_by_stimulus:
                     expected_shape = (target_area.n,)
                     if target_name not in self.connectomes_by_stimulus[sym]:
                          self.connectomes_by_stimulus[sym][target_name] = np.zeros(expected_shape, dtype=np.float32)
                     else:
                          c = self.connectomes_by_stimulus[sym][target_name]
                          if c.shape != expected_shape:
                              c = np.resize(c, expected_shape)
                              self.connectomes_by_stimulus[sym][target_name] = c

        print("DFA Finalized using Robust Manual Wiring.")

    def run_inference(self, input_str):
        """
        Runs the DFA on input string.
        Uses manual projection logic to bypass brain.py's dynamic assembly assumptions
        and ensure 100% deterministic switching with asymmetric connectomes.
        """
        history = []
        current_state = 'q0'
        
        # Reset Areas
        for name, area in self.area_by_name.items():
            area.winners = []
            if name == self.trans_area_name:
                # Keep w=total for Transitions so we can address all neurons
                area.w = self.total_t_neurons
            else:
                area.w = self.k
                
        self.activate(current_state, 0)
        history.append(current_state)
        print(f"Start State: {current_state}")
        
        for i, char in enumerate(input_str):
            print(f"Processing symbol: {char} (Current: {current_state})")
            
            # --- Phase 1: Activate Transition (MANUAL PROJECTION) ---
            # Calculates inputs to ALL neurons in Transitions area
            # inputs = Weights_from_State + Weights_from_Symbol
            t_area = self.area_by_name[self.trans_area_name]
            inputs = np.zeros(t_area.n, dtype=np.float32) 
            
            # Input from State
            if current_state in self.connectomes and self.trans_area_name in self.connectomes[current_state]:
                W_state = self.connectomes[current_state][self.trans_area_name]
                s_winners = self.area_by_name[current_state].winners
                for w_idx in s_winners:
                    inputs += W_state[w_idx]
            
            # Input from Symbol
            if char in self.connectomes_by_stimulus and self.trans_area_name in self.connectomes_by_stimulus[char]:
                 W_sym = self.connectomes_by_stimulus[char][self.trans_area_name]
                 inputs += W_sym
            
            # Winner Take All (k-WTA)
            if self.k < len(inputs):
                potential_winners = np.argpartition(inputs, -self.k)[-self.k:]
            else:
                potential_winners = np.arange(len(inputs))
            
            t_area.winners = potential_winners.tolist()
            
            # Decode Transition (for verification/logging)
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
                 print(f"  -> Activated Rule: ({q_prev}, {sym_prev}) -> {q_target} (votes={max_votes})")
                 next_state_expected = q_target
            else:
                 print(f"  -> Unknown Activation or None")
                 return False, history
            
            # --- Phase 2: Activate Next State (MANUAL PROJECTION) ---
            # Use manual projection again to avoid brain.py broadcasting/shaping errors
            # Project Transition Winners -> All States
            self.area_by_name[current_state].winners = []
            
            state_scores = {}
            for s in self.states:
                s_area = self.area_by_name[s]
                s_inputs = np.zeros(s_area.n, dtype=np.float32)
                
                if self.trans_area_name in self.connectomes and s in self.connectomes[self.trans_area_name]:
                    W_trans = self.connectomes[self.trans_area_name][s]
                    for w_idx in t_area.winners:
                        s_inputs += W_trans[w_idx]
                
                # Identify winners for this state
                if self.k < len(s_inputs):
                    s_winners = np.argpartition(s_inputs, -self.k)[-self.k:]
                else:
                    s_winners = np.arange(len(s_inputs))
                    
                # Check overlap with Canonical Assembly (0..k)
                # Valid states will have high activation in 0..k
                canonical_strength = np.sum(s_inputs[0:self.k])
                state_scores[s] = canonical_strength
                if canonical_strength > 0:
                     s_area.winners = s_winners.tolist()
                else:
                     s_area.winners = []

            # Determine which state won the global competition
            best_s = None
            best_score = 0
            for s, score in state_scores.items():
                if score > best_score:
                    best_score = score
                    best_s = s
            
            if best_s:
                 print(f"  -> Moved to State: {best_s}")
                 history.append(best_s)
                 current_state = best_s
                 # Reinforce to canonical form
                 self.activate(current_state, 0)
            else:
                 print("Failed to activate next state.")
                 return False, history

        is_accept = current_state in self.accept_states
        print(f"Result: {'ACCEPTED' if is_accept else 'REJECTED'} (Path: {history})")
        return is_accept, history
