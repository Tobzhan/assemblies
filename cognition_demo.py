
import brain
import brain_util as bu
import numpy as np

# Patch Brain class
brain.Brain.areas = property(lambda self: self.area_by_name)

class CognitiveDemo:
    def __init__(self):
        self.n = 10000
        self.k = 100
        self.p = 0.05
        self.beta = 0.1
        
        # Initialize Brain with one main "Concept Area" and an "Input Area" (implicit)
        self.b = brain.Brain(self.p, save_winners=True)
        self.b.add_area("Concept", self.n, self.k, self.beta)
        
        # Dictionary to store learned concepts { "Name": set(neuron_indices) }
        self.concepts = {}
        
    def teach_concept(self, name):
        print(f"\n[Teacher] Teaching concept: '{name}'...")
        # Add a specific stimulus for this concept
        stim_name = f"stim_{name}"
        self.b.add_stimulus(stim_name, self.k)
        
        # Project Stimulus -> Concept Area to form an assembly
        # 1. Initial fire
        self.b.project({stim_name: ["Concept"]}, {})
        # 2. Recurrent fire to stabilize
        for _ in range(15):
            self.b.project({stim_name: ["Concept"]}, {"Concept": ["Concept"]})
            
        # Save the resulting assembly as the "Meaning" of this concept
        self.concepts[name] = set(self.b.areas["Concept"].winners)
        print(f"[Brain] Learned '{name}'. Support size: {self.b.areas['Concept'].w}")

    def create_association(self, name1, name2):
        print(f"\n[Teacher] Creating Association: {name1} <---> {name2} ...")
        # To associate, we fire both stimuli together
        stim1 = f"stim_{name1}"
        stim2 = f"stim_{name2}"
        
        # Fire both inputs into Concept area simultaneously
        # This causes the two assemblies to "merge" or wire together
        for _ in range(20):
            self.b.project({stim1: ["Concept"], stim2: ["Concept"]}, {"Concept": ["Concept"]})
            
        # Update our definition of the concepts? 
        # Actually, in a pure association, the original assemblies might shift slightly 
        # or just develop strong links. Let's keep original definitions to see if they naturally trigger each other.
        print(f"[Brain] {name1} and {name2} are now wired together.")

    def think_about(self, name):
        print(f"\n[Experiment] Thinking about '{name}' (Stimulus applied)...")
        stim_name = f"stim_{name}"
        # Fire just this stimulus
        for _ in range(5):
             self.b.project({stim_name: ["Concept"]}, {"Concept": ["Concept"]})

    def teach_math_concepts(self, numbers):
        print(f"\n[Teacher] Teaching Math Concepts: {numbers}...")
        for n in numbers:
            if str(n) not in self.concepts:
                self.teach_concept(str(n))

    def teach_math_fact(self, a, b, result):
        print(f"\n[Teacher] Wiring Operation: {a} + {b} = {result}")
        # Assuming concepts are already taught
        
        stim_a = f"stim_{a}"
        stim_b = f"stim_{b}"
        target_assembly = list(self.concepts[str(result)])
        
        print(f"  Wiring {a} + {b} -> {result}...")
        for _ in range(20):
            self.b.areas["Concept"].winners = target_assembly
            self.b.areas["Concept"].fix_assembly() 
            self.b.project({stim_a: ["Concept"], stim_b: ["Concept"]}, {"Concept": ["Concept"]})
            self.b.areas["Concept"].unfix_assembly()
            
    def test_math(self, a, b):
        print(f"\n[Math Test] What is {a} + {b}?")
        stim_a = f"stim_{a}"
        stim_b = f"stim_{b}"
        
        # Fire A + B
        for _ in range(5):
             self.b.project({stim_a: ["Concept"], stim_b: ["Concept"]}, {"Concept": ["Concept"]})
             
        # Read Mind
        self.read_mind()

    def read_mind(self):
        print("\n[Mind Reader] Scanning Brain State...")
        current_active = set(self.b.areas["Concept"].winners)
        
        found_thoughts = []
        for name, assembly in self.concepts.items():
            # Calculate overlap
            ov = bu.overlap(current_active, assembly, percentage=True)
            if ov > 0.1: # Only report if > 10% overlap
                found_thoughts.append((name, ov))
        
        found_thoughts.sort(key=lambda x: x[1], reverse=True)
        
        if not found_thoughts:
            print("  State: Confused (No recognizable concepts active)")
        else:
            print("  State: Maximum Likelihood Decoding:")
            for name, ov in found_thoughts:
                print(f"   - {name}: {ov:.1%}")

def main():
    demo = CognitiveDemo()
    
    # 1. Teach basic concepts
    demo.teach_concept("Apple")
    demo.teach_concept("Banana")
    demo.teach_concept("Monkey")
    
    # 2. Test initial memory
    demo.think_about("Apple")
    demo.read_mind() # Should correspond to Apple
    
    # 3. Create Association: Monkey likes Banana
    # (We associate them so that firing Monkey triggers Banana nodes too)
    demo.create_association("Monkey", "Banana")
    
    # 4. Test Association
    print("\n--- Testing Association (Recall) ---")
    print("Asking brain to think about 'Monkey' ONLY...")
    
    # Only fire Monkey stimulus. 
    # If association worked, Banana assembly should also light up (turn on).
    demo.think_about("Monkey") 
    
    # Check what is active. 
    # We expect high overlap with Monkey (obviously) AND significant overlap with Banana.
    demo.read_mind()

    # 5. Math Test
    print("\n--- Testing Math (Associative) ---")
    
    # 5a. Teach Numbers individually
    demo.teach_math_concepts([1, 2, 3])
    
    # 5b. Teach Fact
    demo.teach_math_fact(1, 1, 2)
    
    # 5c. Test
    demo.test_math(1, 1)

if __name__ == "__main__":
    main()
