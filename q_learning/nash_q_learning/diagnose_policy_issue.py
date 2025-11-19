"""
Diagnose why evader policy is incorrect at states (0,0,2,4) and (0,0,3,4).
"""

import numpy as np
import json
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nash_q_learning import solve_nash_equilibrium_lp
from competitive_env_5x5 import CompetitiveEnv5x5

def analyze_state(q_values, state_name, pursuer_pos, evader_pos):
    """Analyze a state's Q-values and Nash equilibrium."""
    print(f"\n{'='*60}")
    print(f"State: {state_name}")
    print(f"Pursuer at: {pursuer_pos}, Evader at: {evader_pos}")
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
    
    # Get argmax actions
    pursuer_argmax = np.argmax(pursuer_policy)
    evader_argmax = np.argmax(evader_policy)
    
    print(f"\nInterpretation:")
    print(f"  Pursuer's best action: {action_names[pursuer_argmax]} (probability: {pursuer_policy[pursuer_argmax]:.4f})")
    print(f"  Evader's best action: {action_names[evader_argmax]} (probability: {evader_policy[evader_argmax]:.4f})")
    
    # Check what evader should do
    env = CompetitiveEnv5x5()
    goal = env.evader_goal
    print(f"\nExpected behavior:")
    print(f"  Goal is at: {goal}")
    print(f"  Evader is at: {evader_pos}")
    print(f"  Distance to goal: {abs(evader_pos[0] - goal[0]) + abs(evader_pos[1] - goal[1])}")
    
    # Check what happens if evader goes up
    if evader_pos[0] > goal[0]:  # Evader is below goal
        print(f"  Evader should go UP (action 0) to reach goal!")
        if evader_argmax != 0:
            print(f"  ❌ ERROR: Policy says go {action_names[evader_argmax]}, but should go UP!")
        else:
            print(f"  ✅ Policy correctly says go UP")
    
    # Check evader payoffs for each action
    print(f"\nEvader payoffs (negative of pursuer payoffs, since zero-sum):")
    evader_payoffs = -M.T  # Transpose and negate for evader perspective
    for evader_action in range(4):
        expected_payoff = pursuer_policy @ evader_payoffs[evader_action]
        print(f"  {action_names[evader_action]}: {expected_payoff:.4f} (when pursuer plays Nash)")
    
    return pursuer_policy, evader_policy, value

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Diagnose policy issues')
    parser.add_argument('--q-table', type=str, 
                       default='nash_q_learning_results_5x5_reward20/final_q_table.json',
                       help='Path to final_q_table.json')
    args = parser.parse_args()
    
    # Load Q-table
    q_table_path = args.q_table
    if not os.path.exists(q_table_path):
        print(f"Q-table not found at {q_table_path}")
        print("Please provide the correct path to final_q_table.json")
        return
    
    with open(q_table_path, 'r') as f:
        q_table = json.load(f)
    
    # Analyze problematic states
    # State (0,0,2,4): pursuer at (0,0), evader at (2,4)
    state1_key = "0_0_2_4"
    state2_key = "0_0_3_4"
    
    if state1_key in q_table:
        pursuer_pos = (0, 0)
        evader_pos = (2, 4)
        analyze_state(q_table[state1_key], state1_key, pursuer_pos, evader_pos)
    
    if state2_key in q_table:
        pursuer_pos = (0, 0)
        evader_pos = (3, 4)
        analyze_state(q_table[state2_key], state2_key, pursuer_pos, evader_pos)
    
    # Also check a few more states where evader is close to goal
    print(f"\n{'='*60}")
    print("Checking other states where evader is close to goal...")
    print(f"{'='*60}")
    
    for evader_y in [1, 2, 3, 4]:
        for evader_x in [4]:
            if evader_y == 0 and evader_x == 4:
                continue  # Skip goal itself
            state_key = f"0_0_{evader_y}_{evader_x}"
            if state_key in q_table:
                pursuer_pos = (0, 0)
                evader_pos = (evader_y, evader_x)
                print(f"\nState {state_key}:")
                q_values = q_table[state_key]
                pursuer_policy, evader_policy, value = solve_nash_equilibrium_lp(
                    np.array(q_values).reshape(4, 4))
                evader_argmax = np.argmax(evader_policy)
                action_names = ['Up', 'Down', 'Left', 'Right']
                distance = abs(evader_pos[0] - 0) + abs(evader_pos[1] - 4)
                print(f"  Evader at {evader_pos}, goal at (0,4), distance={distance}")
                print(f"  Evader policy: {evader_policy}")
                print(f"  Evader action: {action_names[evader_argmax]}")
                if evader_y > 0 and evader_argmax != 0:
                    print(f"  ❌ Should go UP but policy says {action_names[evader_argmax]}")

if __name__ == "__main__":
    main()

