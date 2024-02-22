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

    def is_terminal_state(self, state):
        return state == self.goal_state

    def get_transition_states_and_probs(self, state, action):
        """
        Given a state and an action, return a list of (next_state, probability) pairs.
        """
        # Assuming deterministic environment for simplicity
        next_state, _ = self.move(state, action)
        return [(next_state, self.transition_prob)]

    def calculate_q_value(self, V, state, action):
        """
        Calculate the Q-value for a state-action pair given the value function.
        """
        transition_states_and_probs = self.get_transition_states_and_probs(state, action)

        q_value = sum(prob * (self.rewards.get(next_state, -1) + self.gamma * V[next_state])
                      for next_state, prob in transition_states_and_probs)

        return q_value

    def value_iteration(self, threshold=0.001):
        """
        Performs value iteration over the entire grid until the value function converges.
        """
        # Initialize value function to zeros
        V = {s: 0 for s in self.states}

        while True:
            delta = 0
            for state in self.states:
                print(state)
                print(delta)
                if state in self.obstacles or state == self.goal_state:
                    continue  # No need to update value for obstacles or goal state

                v = V[state]  # Store the old value
                V[state] = max([self.calculate_q_value(V, state, action) for action in
                                self.actions])  # Update to the new value

                delta = max(delta, abs(v - V[state]))  # Check the change in value

            if delta < threshold:  # Check for convergence
                break

        return V


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





def plot_grid(mdp, V, current_state):
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

    # Annotating each cell with the value of the state
    for state in mdp.states:
        value_text = "{:.2f}".format(V[state])
        plt.text(state[1], state[0], value_text, ha='center', va='center', color='black')

    plt.title("Current State: {} | Value: {:.2f}".format(current_state, V[current_state]))
    plt.draw()
    plt.pause(0.4)
    plt.clf()


# Initialize MDP and current state
mdp = ComplexGridWorldMDP()

# Perform Value Iteration
V = mdp.value_iteration()

# Derive policy from value function
policy = {state: max(mdp.actions, key=lambda action: mdp.calculate_q_value(V, state, action))
          for state in mdp.states if state not in mdp.obstacles and state != mdp.goal_state}

current_state = mdp.start_state

plt.figure(figsize=(5, 5))

while not mdp.is_terminal_state(current_state):
    plot_grid(mdp, V, current_state)  # Plot the current state of the grid

    # Follow the derived policy
    action = policy[current_state]
    next_state, reward = mdp.move(current_state, action)
    current_state = next_state

plot_grid(mdp, V, current_state)  # Plot final state
plt.show()
