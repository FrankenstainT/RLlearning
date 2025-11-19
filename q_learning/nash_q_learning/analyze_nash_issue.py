"""
Analyze why Nash equilibrium policies appear suboptimal.
"""

import numpy as np
import json
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nash_q_learning import solve_nash_equilibrium_lp

def analyze_state(q_values, state_name):
    """Analyze a state's Q-values and Nash equilibrium."""
    print(f"\n{'='*60}")
    print(f"State: {state_name}")
    print(f"{'='*60}")
    
    # Convert Q-values to 4x4 matrix
    M = np.zeros((4, 4))
    action_names = ['Up', 'Down', 'Left', 'Right']
    
    print("\nQ-value matrix (Pursuer payoffs):")
    print("        Evader: Up    Down   Left   Right")
    for pursuer_action in range(4):
        row = []
        for evader_action in range(4):
            joint_idx = pursuer_action * 4 + evader_action
            M[pursuer_action, evader_action] = q_values[joint_idx]
            row.append(f"{q_values[joint_idx]:7.3f}")
        print(f"Pursuer {action_names[pursuer_action]:4s}: [{', '.join(row)}]")
    
    # Solve Nash equilibrium
    pursuer_policy, evader_policy, value = solve_nash_equilibrium_lp(M)
    
    print(f"\nNash Equilibrium:")
    print(f"  Pursuer policy: {pursuer_policy}")
    print(f"  Evader policy: {evader_policy}")
    print(f"  Value (pursuer's expected payoff): {value:.4f}")
    
    # Check if policies are pure or mixed
    pursuer_argmax = np.argmax(pursuer_policy)
    evader_argmax = np.argmax(evader_policy)
    
    print(f"\nInterpretation:")
    print(f"  Pursuer's best action: {action_names[pursuer_argmax]} (probability: {pursuer_policy[pursuer_argmax]:.4f})")
    print(f"  Evader's best action: {action_names[evader_argmax]} (probability: {evader_policy[evader_argmax]:.4f})")
    
    # Verify Nash equilibrium conditions
    print(f"\nNash Equilibrium Verification:")
    pursuer_payoffs = M @ evader_policy
    evader_payoffs = pursuer_policy @ M
    
    print(f"  Pursuer payoffs for each action: {pursuer_payoffs}")
    print(f"  Max pursuer payoff: {np.max(pursuer_payoffs):.4f}")
    print(f"  Nash value: {value:.4f}")
    print(f"  Difference: {np.max(pursuer_payoffs) - value:.6f}")
    
    print(f"  Evader payoffs (from pursuer's perspective, evader wants to minimize): {evader_payoffs}")
    print(f"  Min evader payoff: {np.min(evader_payoffs):.4f}")
    print(f"  Nash value: {value:.4f}")
    print(f"  Difference: {value - np.min(evader_payoffs):.6f}")
    
    # Check if this is truly a Nash equilibrium
    if abs(np.max(pursuer_payoffs) - value) < 1e-6 and abs(value - np.min(evader_payoffs)) < 1e-6:
        print(f"  ✓ Nash equilibrium conditions satisfied")
    else:
        print(f"  ✗ Nash equilibrium conditions NOT satisfied!")
        print(f"     This suggests the Q-values may not represent a proper zero-sum game.")
    
    return pursuer_policy, evader_policy, value

def main():
    # Load Q-table
    q_table_path = "nash_q_learning_results.3-.001/final_q_table.json"
    with open(q_table_path, 'r') as f:
        q_table = json.load(f)
    
    # Analyze problematic states
    state1 = "1_0_3_1"  # Pursuer at (1,0), Evader at (3,1)
    state2 = "1_1_0_2"  # Pursuer at (1,1), Evader at (0,2)
    
    if state1 in q_table:
        analyze_state(q_table[state1], state1)
    
    if state2 in q_table:
        analyze_state(q_table[state2], state2)
    
    print(f"\n{'='*60}")
    print("EXPLANATION OF THE ISSUE:")
    print(f"{'='*60}")
    print("""
The suboptimal policies are likely due to one or more of the following:

1. **Q-values haven't converged**: The Q-values may not represent the true 
   Nash equilibrium values of the game. This can happen if:
   - States are visited too infrequently
   - Learning rate is too high
   - Not enough training episodes
   - Q-values initialized to 0 and never properly updated

2. **Nash Q-Learning convergence requirements not met**: Nash Q-Learning requires:
   - All state-action pairs visited infinitely often
   - Learning rate must satisfy: sum(alpha) = inf, sum(alpha^2) < inf
   - The game must have certain properties (e.g., unique Nash equilibrium)

3. **Single Q-table limitation**: Using a single Q-table from pursuer's perspective
   assumes the game is zero-sum, but the Q-values must correctly represent this.
   If Q-values are wrong, the Nash equilibrium computed from them will also be wrong.

4. **Mixed strategy interpretation**: The Nash equilibrium might be mixed (probabilistic),
   but when we take argmax, we're selecting a pure strategy that may not be optimal
   if the opponent can exploit it.

5. **The fundamental issue**: If the evader's strategy can be exploited (as you observed),
   then either:
   - The Q-values are incorrect (not converged)
   - The Nash equilibrium solver is finding a local optimum
   - The game structure doesn't match what we're assuming

To fix this, you should:
- Check visit frequencies for these states
- Verify Q-values have converged (TD error near zero)
- Consider using separate Q-tables for each player
- Ensure sufficient exploration and training time
- Verify the Nash equilibrium conditions are actually satisfied
""")

if __name__ == "__main__":
    main()

