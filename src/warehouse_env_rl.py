import random
from typing import Tuple, List


class WarehouseMDP:
    """A 4x4 warehouse MDP with stochastic transitions.

    The grid is indexed by (row, col) from (0, 0) in the top-left to (3, 3)
    in the bottom-right. Actions are encoded as integers:
        0 = up
        1 = right
        2 = down
        3 = left

    Transition probabilities:
        intended direction: 0.8
        left of intended: 0.1
        right of intended: 0.1

    The episode ends when the agent reaches the goal state.
    """

    ACTIONS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
    ACTION_NAMES = {0: "up", 1: "right", 2: "down", 3: "left"}

    def __init__(self, grid_size: int = 4, start: Tuple[int, int] = (3, 0), goal: Tuple[int, int] = (0, 3), hazards: Tuple[Tuple[int, int], ...] = ((1, 3),)):
        self.grid_size = grid_size
        self.start = start
        self.goal = goal
        self.hazards = set(hazards)
        self.terminal_states = {goal}

    def is_terminal(self, state: Tuple[int, int]) -> bool:
        return state in self.terminal_states

    def _apply_action(self, state: Tuple[int, int], action: int) -> Tuple[int, int]:
        dr, dc = self.ACTIONS[action]
        next_row = max(0, min(self.grid_size - 1, state[0] + dr))
        next_col = max(0, min(self.grid_size - 1, state[1] + dc))
        return (next_row, next_col)

    def _stochastic_action(self, action: int) -> int:
        if action not in self.ACTIONS:
            raise ValueError(f"Invalid action: {action}. Valid actions are 0,1,2,3.")

        rnd = random.random()
        if rnd < 0.8:
            return action
        if rnd < 0.9:
            return (action - 1) % 4
        return (action + 1) % 4

    def step(self, state: Tuple[int, int], action: int) -> Tuple[Tuple[int, int], float, bool]:
        """Take one step from the given state using the given action.

        Args:
            state: current grid location as (row, col)
            action: integer action, 0=up, 1=right, 2=down, 3=left

        Returns:
            next_state: resulting grid location after stochastic transition
            reward: reward received for the step
            done: True if the next_state is terminal
        """
        if self.is_terminal(state):
            return state, 0.0, True

        chosen_action = self._stochastic_action(action)
        next_state = self._apply_action(state, chosen_action)
        done = self.is_terminal(next_state)
        
        if next_state in self.hazards:
            reward = -10.0
        elif done:
            reward = 0.0
        else:
            reward = -1.0
        
        return next_state, reward, done

    def reset(self) -> Tuple[int, int]:
        """Reset environment to the start state."""
        return self.start

    def render(self, state: Tuple[int, int]) -> str:
        """Return a string representation of the current grid state."""
        rows: List[str] = []
        for r in range(self.grid_size):
            cells: List[str] = []
            for c in range(self.grid_size):
                if (r, c) == state:
                    cells.append("A")
                elif (r, c) == self.goal:
                    cells.append("G")
                elif (r, c) in self.hazards:
                    cells.append("H")
                else:
                    cells.append(".")
            rows.append(" ".join(cells))
        return "\n".join(rows)
