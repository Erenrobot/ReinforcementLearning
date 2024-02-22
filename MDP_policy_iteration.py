import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


class ComplexGridWorldMDP:
    def __init__(self):
        self.size = 5
        self.states = [(i, j) for i in range(self.size) for j in range(self.size)]
        self.start_state = (0, 0)
        self.goal_state = (4, 4)
        self.obstacles = [(1, 1), (1, 3), (2, 2), (3, 0), (3, 3)]
        self.actions = ["up", "down", "left", "right"]
        self.rewards = {self.goal_state: 10}
        for obs in self.obstacles:
            self.rewards[obs] = -10  # high penalty for hitting an obstacle
        self.transition_prob = 1  # deterministic for simplicity
        self.gamma = 0.9

    def calculate_q_value(self, V, state, action):
        """
        Calculate the Q-value for a state-action pair given the value function.
        """
        # Assuming deterministic environment for simplicity
        next_state, reward = self.move(state, action)
        q_value = reward + self.gamma * V[next_state]  # Bellman equation
        return q_value
    def is_terminal_state(self, state):
        return state == self.goal_state

    def policy_evaluation(self, policy, threshold=0.001):
        """
        Evaluate a policy by calculating the value function for the current policy.
        """
        V = {s: 0 for s in self.states}
        while True:
            delta = 0
            for state in self.states:
                if state in self.obstacles or state == self.goal_state:
                    continue
                v = V[state]
                action = policy[state]
                next_state, reward = self.move(state, action)
                V[state] = reward + self.gamma * V[next_state]  # Corrected line
                delta = max(delta, abs(v - V[state]))
            if delta < threshold:
                break
        return V

    def policy_improvement(self, V, policy):
        """
        Improve the policy based on the current value function.
        """
        policy_stable = True
        new_policy = {}
        for state in self.states:
            if state in self.obstacles or state == self.goal_state:
                continue
            old_action = policy.get(state, None)
            # Choose the action that maximizes the value function
            new_action = max(self.actions, key=lambda action: self.calculate_q_value(V, state, action))
            new_policy[state] = new_action
            if old_action is None or old_action != new_action:
                policy_stable = False
        return new_policy, policy_stable

    def policy_iteration(self):
        """
        Perform policy iteration using policy evaluation and policy improvement.
        """
        # Initialize a random policy
        policy = {state: np.random.choice(self.actions) for state in self.states if
                  state not in self.obstacles and state != self.goal_state}

        while True:
            V = self.policy_evaluation(policy)
            new_policy, policy_stable = self.policy_improvement(V, policy)
            policy = new_policy  # Update the policy to the new policy
            if policy_stable:
                break
        return policy, V
    def move(self, state, action):
        if state in self.obstacles or self.is_terminal_state(state):
            return state, 0

        # Define next state based on action
        next_state = list(state)
        if action == "up" and state[0] > 0:
            next_state[0] -= 1
        elif action == "down" and state[0] < self.size - 1:
            next_state[0] += 1
        elif action == "left" and state[1] > 0:
            next_state[1] -= 1
        elif action == "right" and state[1] < self.size - 1:
            next_state[1] += 1
        next_state = tuple(next_state)

        if next_state in self.obstacles:
            return state, self.rewards.get(state, -1)

        return next_state, self.rewards.get(next_state, -1)

    def get_transition_states_and_probs(self, state, action):
        """
        Given a state and an action, return a list of (next_state, probability) pairs.
        """
        # Assuming deterministic environment for simplicity
        next_state, _ = self.move(state, action)
        return [(next_state, self.transition_prob)]







def plot_grid(mdp, V, policy, current_state):
    grid = np.zeros((mdp.size, mdp.size))
    for obs in mdp.obstacles:
        grid[obs] = -1  # represent obstacles
    grid[mdp.goal_state] = 2  # represent goal state
    grid[current_state] = 1  # represent agent's current state

    cmap = mcolors.ListedColormap(['white', 'blue', 'green', 'red'])
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    plt.imshow(grid, cmap=cmap, norm=norm)
    plt.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    plt.xticks([])
    plt.yticks([])

    # Adding arrows for the policy
    for state in mdp.states:
        if state in mdp.obstacles or state == mdp.goal_state:
            continue
        action = policy.get(state, None)
        if action:
            dx, dy = 0, 0
            if action == "up":
                dy = -0.5
            elif action == "down":
                dy = 0.5
            elif action == "left":
                dx = -0.5
            elif action == "right":
                dx = 0.5
            plt.arrow(state[1], state[0], dx, dy, head_width=0.1, head_length=0.1, fc='k', ec='k')

    plt.title("Current State: {} | Value: {:.2f}".format(current_state, V[current_state]))
    plt.draw()
    plt.pause(0.2)
    plt.clf()


# Initialize MDP and current state
# Initialize MDP and current state
mdp = ComplexGridWorldMDP()

# Perform Policy Iteration
policy, V = mdp.policy_iteration()

current_state = mdp.start_state

plt.figure(figsize=(5, 5))

while not mdp.is_terminal_state(current_state):
    plot_grid(mdp, V, policy, current_state)  # Pass the policy here

    # Follow the derived policy
    action = policy[current_state]
    next_state, reward = mdp.move(current_state, action)
    current_state = next_state

plot_grid(mdp, V, policy, current_state)  # Plot final state with policy
plt.show()
