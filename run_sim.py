
import random
import brain
import brain_util as bu
import matplotlib.pyplot as plt

# Patch Brain class to match logic in simulations.py
# simulations.py expects b.areas["Name"], Brain uses b.area_by_name
brain.Brain.areas = property(lambda self: self.area_by_name)

def run_and_visualize_pattern_completion():
    print("Running Pattern Completion Simulation...")
    print("This simulates 'Pattern Completion':")
    print("1) Learn an assembly in area A by repeated projection from a stimulus.")
    print("2) Silence the stimulus and activate only an alpha-fraction of the learned assembly.")
    print("3) Run recurrent dynamics in A and measure whether activity returns toward the learned assembly.")

    # --- Parameters (tuned to reliably show recovery) ---
    # NOTE: For explicit simulation, n controls memory use ~O(n^2). Keep it modest.
    n = 5000        # neurons in area A (explicit)
    k = 100         # winners per step (assembly size)
    p = 0.01        # connection probability
    beta = 0.05     # plasticity / activation parameter
    alpha = 0.5    # cue fraction of the learned assembly
    learn_rounds = 50
    recover_rounds = 10

    # Reproducibility
    seed = 0

    b = brain.Brain(p, save_winners=True, seed=seed)
    b.add_stimulus("stim", k)

    # Use an *explicit* area so recurrent connections are actually represented,
    # which makes pattern completion much easier to observe.
    b.add_explicit_area("A", n, k, beta)

    print("\nPhase 1: Learning assembly in A ...")
    # First projection: stimulus -> A (initializes activity)
    b.project({"stim": ["A"]}, {})
    print(f"Round 0: {b.areas['A'].num_first_winners} new winners")

    # Subsequent rounds: stimulus -> A plus recurrent A -> A
    for t in range(1, learn_rounds + 1):
        b.project({"stim": ["A"]}, {"A": ["A"]})
        if t <= 5 or t in (10, 20, 30, 40, learn_rounds):
            print(f"Round {t}: {b.areas['A'].num_first_winners} new winners")

    learned_assembly = set(b.areas["A"].winners)
    print(f"Learned assembly size: {len(learned_assembly)} (expected {k})")

    print("\nPhase 2: Pattern completion (no stimulus) ...")
    # IMPORTANT: turn off plasticity during recall so the target assembly doesn't drift.
    b.disable_plasticity = True

    subsample_size = max(1, int(k * alpha))
    cue = random.sample(b.areas["A"].winners, subsample_size)
    b.areas["A"].winners = cue
    print(f"Cueing with alpha={alpha} -> firing subset size {len(cue)}")

    overlaps = []
    rounds = []

    # Round 0 overlap (just the cue)
    overlaps.append(bu.overlap(cue, learned_assembly, percentage=True))
    rounds.append(0)

    for t in range(1, recover_rounds + 1):
        b.project({}, {"A": ["A"]})
        current = b.areas["A"].winners
        ov = bu.overlap(current, learned_assembly, percentage=True)
        print(f"Recovery round {t}: overlap with learned assembly = {ov:.2%}")
        overlaps.append(ov)
        rounds.append(t)

    # Plotting: overlap vs recurrent rounds
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, overlaps, marker="o")
    plt.axhline(y=1.0, linestyle="--", label="Perfect completion (100%)")
    plt.axhline(y=alpha, linestyle="--", label=f"Initial cue (alpha={alpha})")
    plt.title(f"Pattern Completion in Area A (explicit), alpha={alpha}")
    plt.xlabel("Recurrent rounds (A -> A)")
    plt.ylabel("Overlap with learned assembly")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True)

    out_path = "pattern_completion.png"
    print(f"\nSaving plot to '{out_path}' ...")
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    print("Done.")

if __name__ == "__main__":
    run_and_visualize_pattern_completion()
