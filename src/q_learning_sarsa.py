import random
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


class QLearningAgent:
    """Tabular Q-learning agent with epsilon-greedy exploration and epsilon decay."""

    def __init__(
        self,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.01,
    ):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_values: Dict[Tuple[int, int], List[float]] = {}

    def _ensure_state(self, state: Tuple[int, int]) -> None:
        if state not in self.q_values:
            self.q_values[state] = [0.0] * self.n_actions

    def select_action(self, state: Tuple[int, int]) -> int:
        self._ensure_state(state)
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        return int(max(range(self.n_actions), key=lambda a: self.q_values[state][a]))

    def update(
        self,
        state: Tuple[int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int],
        done: bool,
    ) -> None:
        self._ensure_state(state)
        self._ensure_state(next_state)
        best_next = 0.0 if done else max(self.q_values[next_state])
        target = reward + self.gamma * best_next
        td_error = target - self.q_values[state][action]
        self.q_values[state][action] += self.alpha * td_error

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def train(
        self,
        env,
        episodes: int = 500,
        max_steps_per_episode: int = 100,
    ) -> List[float]:
        """Train on the given environment and return total reward per episode."""
        episode_rewards: List[float] = []

        for _ in range(episodes):
            state = env.reset()
            total_reward = 0.0
            done = False
            steps = 0

            while not done and steps < max_steps_per_episode:
                action = self.select_action(state)
                next_state, reward, done = env.step(state, action)
                self.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                steps += 1

            episode_rewards.append(total_reward)
            self.decay_epsilon()

        return episode_rewards

    def get_action_values(self, state: Tuple[int, int]) -> List[float]:
        self._ensure_state(state)
        return self.q_values[state]

    def extract_policy(self, states: List[Tuple[int, int]] = None) -> Dict[Tuple[int, int], int]:
        """Return the greedy policy learned from Q-values.

        If states is provided, produce a policy for every state in the list.
        """
        if states is None:
            states = list(self.q_values.keys())

        policy: Dict[Tuple[int, int], int] = {}
        for state in states:
            self._ensure_state(state)
            policy[state] = int(max(range(self.n_actions), key=lambda a: self.q_values[state][a]))
        return policy


def rolling_average(values: List[float], window: int = 50) -> List[float]:
    if window <= 1:
        return values[:]
    averages: List[float] = []
    cumulative = 0.0
    for idx, value in enumerate(values):
        cumulative += value
        if idx >= window:
            cumulative -= values[idx - window]
        averages.append(cumulative / min(window, idx + 1))
    return averages


def value_iteration(env, gamma: float = 0.99, theta: float = 1e-6):
    states = [(r, c) for r in range(env.grid_size) for c in range(env.grid_size)]
    V = {state: 0.0 for state in states}
    terminal = env.goal

    def _next_state(state, action):
        dr, dc = env.ACTIONS[action]
        next_row = max(0, min(env.grid_size - 1, state[0] + dr))
        next_col = max(0, min(env.grid_size - 1, state[1] + dc))
        return (next_row, next_col)

    def _transitions(state, action):
        if state == terminal:
            return [(1.0, state, 0.0, True)]
        outcomes = []
        for prob, direction in [(0.8, action), (0.1, (action - 1) % 4), (0.1, (action + 1) % 4)]:
            next_state = _next_state(state, direction)
            done = next_state == terminal
            if next_state in env.hazards:
                reward = -10.0
            elif done:
                reward = 0.0
            else:
                reward = -1.0
            outcomes.append((prob, next_state, reward, done))
        return outcomes

    while True:
        delta = 0.0
        for state in states:
            if state == terminal:
                continue
            old_value = V[state]
            action_values = []
            for action in range(4):
                expected = 0.0
                for prob, next_state, reward, done in _transitions(state, action):
                    expected += prob * (reward + gamma * V[next_state])
                action_values.append(expected)
            V[state] = max(action_values)
            delta = max(delta, abs(old_value - V[state]))
        if delta < theta:
            break

    policy = {
        state: int(max(range(4), key=lambda action: sum(
            prob * (reward + gamma * V[next_state])
            for prob, next_state, reward, done in _transitions(state, action)
        )))
        for state in states if state != terminal
    }
    return V, policy


def plot_learning_curves(
    q_rewards: List[float],
    sarsa_rewards: List[float],
    optimal_return: float,
    window: int = 50,
) -> None:
    q_avg = rolling_average(q_rewards, window)
    sarsa_avg = rolling_average(sarsa_rewards, window)

    plt.figure(figsize=(9, 5))
    plt.plot(q_avg, label="Q-learning")
    plt.plot(sarsa_avg, label="SARSA")
    plt.axhline(optimal_return, color="black", linestyle="--", label=f"Optimal expected return = {optimal_return:.2f}")
    plt.xlabel("Episode")
    plt.ylabel(f"Rolling average reward (window={window})")
    plt.title("Q-learning vs SARSA: rolling average reward")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def action_to_vector(action: int, env) -> Tuple[float, float]:
    dr, dc = env.ACTIONS[action]
    return float(dc) * 0.4, float(-dr) * 0.4


def plot_policy_grid(policy: Dict[Tuple[int, int], int], env, title: str, ax) -> None:
    grid_size = env.grid_size
    xs: List[float] = []
    ys: List[float] = []
    us: List[float] = []
    vs: List[float] = []

    for r in range(grid_size):
        for c in range(grid_size):
            state = (r, c)
            if state == env.goal:
                continue
            action = policy.get(state)
            if action is None:
                continue
            x = float(c)
            y = float(grid_size - 1 - r)
            u, v = action_to_vector(action, env)
            xs.append(x)
            ys.append(y)
            us.append(u)
            vs.append(v)

    ax.quiver(xs, ys, us, vs, angles="xy", scale_units="xy", scale=1, width=0.015)
    ax.scatter([env.start[1]], [grid_size - 1 - env.start[0]], color="green", s=80, marker="o", label="start")
    ax.scatter([env.goal[1]], [grid_size - 1 - env.goal[0]], color="red", s=80, marker="X", label="goal")
    
    if hasattr(env, 'hazards') and env.hazards:
        hazard_cols = [c for r, c in env.hazards]
        hazard_rows = [grid_size - 1 - r for r, c in env.hazards]
        ax.scatter(hazard_cols, hazard_rows, color="orange", s=100, marker="s", label="hazard")
    
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_xticks(list(range(grid_size)))
    ax.set_yticks(list(range(grid_size)))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", fontsize="small")


def plot_policy_comparison(
    q_policy: Dict[Tuple[int, int], int],
    sarsa_policy: Dict[Tuple[int, int], int],
    optimal_policy: Dict[Tuple[int, int], int],
    env,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    plot_policy_grid(q_policy, env, "Q-learning policy", axes[0])
    plot_policy_grid(sarsa_policy, env, "SARSA policy", axes[1])
    plot_policy_grid(optimal_policy, env, "Optimal policy", axes[2])
    plt.tight_layout()
    plt.show()


def policy_match_report(
    learned_policy: Dict[Tuple[int, int], int],
    optimal_policy: Dict[Tuple[int, int], int],
    states: List[Tuple[int, int]],
) -> Tuple[int, List[Tuple[Tuple[int, int], int, int, bool]]]:
    report: List[Tuple[Tuple[int, int], int, int, bool]] = []
    match_count = 0
    for state in states:
        learned_action = learned_policy.get(state)
        optimal_action = optimal_policy[state]
        matched = learned_action == optimal_action
        report.append((state, learned_action, optimal_action, matched))
        if matched:
            match_count += 1
    return match_count, report


class SarsaAgent(QLearningAgent):
    """Tabular SARSA agent with epsilon-greedy exploration and epsilon decay."""

    def update(
        self,
        state: Tuple[int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int],
        next_action: int,
        done: bool,
    ) -> None:
        self._ensure_state(state)
        self._ensure_state(next_state)
        next_q = 0.0 if done else self.q_values[next_state][next_action]
        target = reward + self.gamma * next_q
        td_error = target - self.q_values[state][action]
        self.q_values[state][action] += self.alpha * td_error

    def train(
        self,
        env,
        episodes: int = 500,
        max_steps_per_episode: int = 100,
    ) -> List[float]:
        """Train with SARSA and return total reward per episode."""
        episode_rewards: List[float] = []

        for _ in range(episodes):
            state = env.reset()
            self._ensure_state(state)
            action = self.select_action(state)
            total_reward = 0.0
            done = False
            steps = 0

            while not done and steps < max_steps_per_episode:
                next_state, reward, done = env.step(state, action)
                next_action = self.select_action(next_state) if not done else 0
                self.update(state, action, reward, next_state, next_action, done)
                state = next_state
                action = next_action
                total_reward += reward
                steps += 1

            episode_rewards.append(total_reward)
            self.decay_epsilon()

        return episode_rewards


def hyperparameter_sweep(
    learning_rates: List[float] = [0.01, 0.1, 0.5],
    epsilon_decays: List[float] = [0.99, 0.995, 0.999],
    episodes: int = 500,
    max_steps_per_episode: int = 100,
) -> None:
    """Run a grid search over learning rates and epsilon decays."""
    try:
        from warehouse_env_rl import WarehouseMDP
    except ImportError:
        print("Could not import WarehouseMDP. Please run this script from the repository root.")
        return

    fig, axes = plt.subplots(len(learning_rates), len(epsilon_decays), figsize=(15, 12))
    
    for i, alpha in enumerate(learning_rates):
        for j, eps_decay in enumerate(epsilon_decays):
            env = WarehouseMDP()
            
            q_agent = QLearningAgent(
                n_actions=4,
                alpha=alpha,
                gamma=0.99,
                epsilon=1.0,
                epsilon_decay=eps_decay,
                min_epsilon=0.01,
            )
            sarsa_agent = SarsaAgent(
                n_actions=4,
                alpha=alpha,
                gamma=0.99,
                epsilon=1.0,
                epsilon_decay=eps_decay,
                min_epsilon=0.01,
            )
            
            q_rewards = q_agent.train(env, episodes=episodes, max_steps_per_episode=max_steps_per_episode)
            sarsa_rewards = sarsa_agent.train(env, episodes=episodes, max_steps_per_episode=max_steps_per_episode)
            
            q_avg = rolling_average(q_rewards, window=50)
            sarsa_avg = rolling_average(sarsa_rewards, window=50)
            
            ax = axes[i, j]
            ax.plot(q_avg, label="Q-learning", linewidth=2)
            ax.plot(sarsa_avg, label="SARSA", linewidth=2)
            ax.set_title(f"α={alpha}, decay={eps_decay}")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Rolling avg reward")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle("Hyperparameter Sweep: Learning Rate × Epsilon Decay", fontsize=14, y=1.00)
    plt.show()


def compare_q_learning_and_sarsa():
    try:
        from warehouse_env_rl import WarehouseMDP
    except ImportError:
        print("Could not import WarehouseMDP. Please run this script from the repository root.")
        return

    env = WarehouseMDP()
    q_agent = QLearningAgent(
        n_actions=4,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.01,
    )
    sarsa_agent = SarsaAgent(
        n_actions=4,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.01,
    )

    episodes = 500
    max_steps_per_episode = 100

    q_rewards = q_agent.train(env, episodes=episodes, max_steps_per_episode=max_steps_per_episode)
    sarsa_rewards = sarsa_agent.train(env, episodes=episodes, max_steps_per_episode=max_steps_per_episode)
    optimal_values, optimal_policy = value_iteration(env, gamma=0.99)
    optimal_return = optimal_values[env.start]

    states = [
        (r, c)
        for r in range(env.grid_size)
        for c in range(env.grid_size)
        if (r, c) != env.goal
    ]
    q_policy = q_agent.extract_policy(states)
    sarsa_policy = sarsa_agent.extract_policy(states)

    q_matches, q_report = policy_match_report(q_policy, optimal_policy, states)
    sarsa_matches, sarsa_report = policy_match_report(sarsa_policy, optimal_policy, states)

    print(f"Trained Q-learning and SARSA for {episodes} episodes each.")
    print(f"Optimal expected return from value iteration: {optimal_return:.2f}")
    print(f"Q-learning matches optimal actions in {q_matches}/{len(states)} states.")
    print(f"SARSA matches optimal actions in {sarsa_matches}/{len(states)} states.")
    print("\nState-level action comparison:\n")
    print("State  Q  SARSA  Optimal  Q_match  SARSA_match")
    for state, q_action, optimal_action, q_match in q_report:
        sarsa_action = next(item[1] for item in sarsa_report if item[0] == state)
        sarsa_match = next(item[3] for item in sarsa_report if item[0] == state)
        print(
            f"{state}  {q_action}    {sarsa_action}    {optimal_action}      {q_match}      {sarsa_match}"
        )

    plot_learning_curves(q_rewards, sarsa_rewards, optimal_return, window=50)
    plot_policy_comparison(q_policy, sarsa_policy, optimal_policy, env)


if __name__ == "__main__":
    hyperparameter_sweep(
        learning_rates=[0.01, 0.1, 0.5],
        epsilon_decays=[0.99, 0.995, 0.999],
        episodes=500,
        max_steps_per_episode=100,
    )
