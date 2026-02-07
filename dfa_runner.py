from dfa_brain import DFABrain

def run_regex_demo():
    print("=== DFA Regex Demo: ab* ===")
    
    # 1. Setup Brain
    # States: q0 (start), q1 (accept), q_err (reject)
    # Alphabet: a, b
    # Logic:
    #   q0 --a--> q1
    #   q1 --b--> q1
    #   q0 --b--> q_err
    #   q1 --a--> q_err
    #   q_err --*--> q_err
    
    # Large beta to ensure one-shot learning works adequately
    brain = DFABrain(p=0.01, beta=0.8) 
    
    brain.add_state("q0", is_start=True)
    brain.add_state("q1", is_accept=True)
    brain.add_state("q_err")
    
    brain.add_symbol("a")
    brain.add_symbol("b")
    
    brain.set_transition("q0", "a", "q1")
    brain.set_transition("q1", "b", "q1")
    
    # Error cases
    brain.set_transition("q0", "b", "q_err")
    brain.set_transition("q1", "a", "q_err")
    brain.set_transition("q_err", "a", "q_err")
    brain.set_transition("q_err", "b", "q_err")
    
    # 2. Finalize (Train)
    brain.finalize_machine()
    
    # 3. Test Cases
    test_strings = [
        "a",      # Valid
        "ab",     # Valid
        "abb",    # Valid
        "b",      # Invalid
        "ba",     # Invalid
        "aa",     # Invalid
        "abbb",   # Valid
        "aba",    # Invalid
        "abbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "abbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba",
        "abbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbab",
        "aboba",
        "q",
    ]
    
    for s in test_strings:
        print(f"\n--- Testing string '{s}' ---")
        accepted, history = brain.run_inference(s)
        result_str = "ACCEPTED" if accepted else "REJECTED"
        print(f"Result: {result_str} (Path: {history})")
        
        # Validation logic for ab*
        # Must start with 'a', then zero or more 'b's.
        is_valid_logic = (len(s) > 0) and (s[0] == 'a') and (all(c == 'b' for c in s[1:]))
        
        expected_str = "ACCEPTED" if is_valid_logic else "REJECTED"
        if result_str == expected_str:
            print(">> VERIFICATION PASSED")
        else:
            print(f">> VERIFICATION FAILED (Expected {expected_str})")

if __name__ == "__main__":
    run_regex_demo()
