
import brain
import simulations
import brain_util as bu
import matplotlib.pyplot as plt
import numpy as np
import argparse
import copy

# Patch Brain class to match logic in simulations.py
brain.Brain.areas = property(lambda self: self.area_by_name)

def run_projection(args):
    print("Running Projection Simulation...")
    print("Simulating convergence of assembly size over 50 rounds for different Beta values.")
    
    # We use a smaller n for speed in this interactive demo, but keep proportions
    n = 10000 
    k = 317 # Standard sqrt(n) approx
    p = 0.05
    t = 50
    
    betas = [0.25, 0.1, 0.05, 0.01]
    results = {}
    
    plt.figure(figsize=(10, 6))
    
    for beta in betas:
        print(f"  Simulating beta={beta}...")
        b = brain.Brain(p, save_winners=True)
        b.add_stimulus("stim", k)
        b.add_area("A", n, k, beta)
        b.project({"stim":["A"]},{})
        for i in range(t-1):
            b.project({"stim":["A"]},{"A":["A"]})
            
        results[beta] = b.areas["A"].saved_w
        plt.plot(range(len(results[beta])), results[beta], label=f'beta={beta}')

    plt.title(f'Projection Convergence (n={n}, k={k}, p={p})')
    plt.xlabel('Round')
    plt.ylabel('Assembly Size (w)')
    plt.legend()
    plt.grid(True)
    plt.savefig('projection_results.png')
    print("Saved plot to 'projection_results.png'")

def run_merge(args):
    print("Running Merge Simulation...")
    print("Simulating merge of two assemblies A and B into C.")
    
    n = 10000
    k = 100
    p = 0.01
    beta = 0.1
    max_t = 30
    
    b = brain.Brain(p)
    b.add_stimulus("stimA", k)
    b.add_stimulus("stimB", k)
    b.add_area("A", n, k, beta)
    b.add_area("B", n, k, beta)
    b.add_area("C", n, k, beta)
    
    # 1. Form A
    print("  Forming assembly A...")
    b.project({"stimA":["A"]}, {})
    for _ in range(10): 
        b.project({"stimA":["A"]}, {"A":["A"]})
        
    # 2. Form B
    print("  Forming assembly B...")
    b.project({"stimB":["B"]}, {})
    for _ in range(10):
        b.project({"stimB":["B"]}, {"B":["B"]})
        
    # 3. Merge into C
    print("  Merging A and B into C...")
    # Fire A and B together into C
    
    c_sizes = []
    
    # Initial firing into C from A,B
    # Note: connectomes are established on the fly
    b.project({"stimA":["A"], "stimB":["B"]}, {"A":["A", "C"], "B":["B", "C"]})
    c_sizes.append(b.areas["C"].w)
    
    # Recurrent firing
    for i in range(max_t):
        b.project({"stimA":["A"], "stimB":["B"]}, 
                  {"A":["A", "C"], "B":["B", "C"], "C":["C", "A", "B"]})
        c_sizes.append(b.areas["C"].w)
        
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(c_sizes)), c_sizes, marker='o')
    plt.title(f'Merge Operation (A+B->C) Convergence')
    plt.xlabel('Merge Round')
    plt.ylabel('Size of C')
    plt.grid(True)
    plt.savefig('merge_results.png')
    print("Saved plot to 'merge_results.png'")

def run_association(args):
    print("Running Association Simulation...")
    print("Simulating association between Assembly A and Assembly B via area C.")
    
    n = 10000
    k = 100
    p = 0.05
    beta = 0.1
    perturb_strength = 0.3 # fire 30% of C to recall
    
    b = brain.Brain(p, save_winners=True)
    b.add_stimulus("stimA", k)
    b.add_area("A", n, k, beta)
    b.add_stimulus("stimB", k)
    b.add_area("B", n, k, beta)
    
    # 1. Form Assembly A
    print("  Forming A...")
    b.project({"stimA":["A"]}, {})
    for _ in range(10): b.project({"stimA":["A"]}, {"A":["A"]})
    assembly_A = b.areas["A"].winners
    
    # 2. Form Assembly B
    print("  Forming B...")
    b.project({"stimB":["B"]}, {})
    for _ in range(10): b.project({"stimB":["B"]}, {"B":["B"]})
    assembly_B = b.areas["B"].winners
    
    # 3. Associate A <-> B (Since there is no direct link A-B usually, we cheat slightly 
    #    or use recursive/updates. But standard association in this paper usually implies 
    #    A <-> B direct connection or co-wiring if areas are connected.
    #    Let's check `associate()` in simulations.py... it uses C as an inter-area?
    #    Wait, `associate` in simulations.py uses C. Let's strictly follow that logic.
    
    print("  Using standard simulations.py associate logic (A <-> C <-> B)...")
    b = brain.Brain(p, save_winners=True)
    b.add_stimulus("stimA",k)
    b.add_area("A",n,k,beta)
    b.add_stimulus("stimB",k)
    b.add_area("B",n,k,beta)
    b.add_area("C",n,k,beta)
    
    # Creating assemblies A and B
    b.project({"stimA":["A"],"stimB":["B"]},{})
    for i in range(10):
        b.project({"stimA":["A"],"stimB":["B"]}, {"A":["A"],"B":["B"]})
        
    final_A = b.areas["A"].winners
    final_B = b.areas["B"].winners
    
    print("  Projecting A->C...")
    b.project({"stimA":["A"]},{"A":["A","C"]})
    for i in range(10):
        b.project({"stimA":["A"]}, {"A":["A","C"],"C":["C"]})
        
    print("  Projecting B->C...")
    b.project({"stimB":["B"]},{"B":["B","C"]})
    for i in range(10):
        b.project({"stimB":["B"]}, {"B":["B","C"],"C":["C"]})
        
    print("  Projecting A+B -> C (Association Phase)...")
    b.project({"stimA":["A"],"stimB":["B"]}, {"A":["A","C"],"B":["B","C"]})
    for i in range(10):
        b.project({"stimA":["A"],"stimB":["B"]}, {"A":["A","C"],"B":["B","C"],"C":["C"]})
        
    # Test Association: Fire A -> See if C fires -> See if B fires (Recall)
    print("  Testing Recall A -> C -> B...")
    
    # Just fire stimA -> A
    # Then A -> C
    # Then C -> B
    
    # Step 1: Fire A
    b.project({"stimA":["A"]}, {"A":["A"]})
    # Step 2: Fire A->C (and A->A)
    b.project({"stimA":["A"]}, {"A":["A", "C"], "C":["C"]})
    # Step 3: Fire C->B (and C->C)
    # We turn off stimA to see if A sustains and propagates, or just purely from C
    b.project({}, {"C":["C", "B"], "B":["B"]})
    
    recovered_B = b.areas["B"].winners
    overlap_B = bu.overlap(recovered_B, final_B, percentage=True)
    
    print(f"  Recovered B from A-cue via C: {overlap_B:.2%}")
    
    # Visualization: Bar chart of overlaps
    plt.figure(figsize=(6, 6))
    plt.bar(["Recall B via A->C"], [overlap_B])
    plt.ylim(0, 1.1)
    plt.ylabel("Overlap with original B")
    plt.title("Association Recall w/ Inter-area C")
    plt.grid(True, axis='y')
    plt.savefig('association_results.png')
    print("Saved plot to 'association_results.png'")

def main():
    parser = argparse.ArgumentParser(description="Run Assembly Simulations")
    parser.add_argument('op', choices=['projection', 'merge', 'association'], help='Operation to simulate')
    args = parser.parse_args()
    
    if args.op == 'projection':
        run_projection(args)
    elif args.op == 'merge':
        run_merge(args)
    elif args.op == 'association':
        run_association(args)

if __name__ == "__main__":
    main()
