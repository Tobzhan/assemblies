"""
Generate a demo simulation log for the visualization app.
Run: python generate_demo_log.py
"""

from logged_dfa_brain import LoggedDFABrain


def main():
    brain = LoggedDFABrain(p=0.01, beta=0.8, log_dir="logs")

    brain.add_state("q0", is_start=True)
    brain.add_state("q1", is_accept=True)
    brain.add_state("q_err")

    brain.add_symbol("a")
    brain.add_symbol("b")

    brain.set_transition("q0", "a", "q1")
    brain.set_transition("q1", "b", "q1")
    brain.set_transition("q0", "b", "q_err")
    brain.set_transition("q1", "a", "q_err")
    brain.set_transition("q_err", "a", "q_err")
    brain.set_transition("q_err", "b", "q_err")

    brain.finalize_machine()

    # Run several test strings to produce a rich log
    test_strings = ["a", "ab", "abbb", "b", "ba", "abb"]
    for s in test_strings:
        print(f"\n--- Running '{s}' ---")
        accepted, history = brain.run_inference(s)
        result = "ACCEPTED" if accepted else "REJECTED"
        print(f"Result: {result} | Path: {history}")

    path = brain.save_log()
    print(f"\n✅ Log saved to: {path}")
    print(f"   Total steps logged: {len(brain.logger.steps)}")


if __name__ == "__main__":
    main()
