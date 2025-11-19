"""
Nash DQN for 5x5 Competitive Pursuer-Evader Game
=================================================

Wrapper around the existing Nash DQN implementation, adapted for 5x5 game.
"""

import sys
import os

# Add parent directory to import existing Nash DQN
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nash_competitive.nash_dqn import NashDQN, solve_nash_equilibrium

# Re-export for convenience
solve_nash_equilibrium_lp = solve_nash_equilibrium

# Create alias for 5x5 version
NashDQN5x5 = NashDQN

__all__ = ['NashDQN5x5', 'solve_nash_equilibrium_lp']

