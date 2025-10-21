import numpy as np

def policy_iteration(env, gamma=0.9, theta=1e-4, max_iterations=1000):
    """
    Perform Policy Iteration on a discrete Grid Maze environment.
    Assumes env positions are (row, col) and uses row-major indexing:
        state_index = row * grid_size + col
    Returns optimal policy (n_states x n_actions) and value function V (n_states,).
    """

    n_states = env.grid_size * env.grid_size
    n_actions = env.action_space.n

    # uniform random policy to start
    policy = np.ones([n_states, n_actions]) / n_actions
    V = np.zeros(n_states)

    def state_to_index(pos):
        # pos is (row, col)
        return int(pos[0]) * env.grid_size + int(pos[1])

    # --- Identify terminal states and their fixed values ---
    goal_idx = state_to_index(env.goal_pos)
    bad_idx1 = state_to_index(env.bad1_pos)
    bad_idx2 = state_to_index(env.bad2_pos)
    terminal_states = {goal_idx: 10.0, bad_idx1: -10.0, bad_idx2: -10.0}

    # Precompute transitions: transitions[s][a] = [(prob, next_state_idx, reward), ...]
    transitions = {}
    for row in range(env.grid_size):
        for col in range(env.grid_size):
            s = state_to_index((row, col))
            transitions[s] = {}
            for a in range(n_actions):
                transitions[s][a] = []
                # stochastic movement probabilities
                move_probs = [0.0] * n_actions
                move_probs[a] = 0.7
                for pa in [(a + 1) % 4, (a + 3) % 4]:
                    move_probs[pa] = 0.15

                for aa, p in enumerate(move_probs):
                    d_row, d_col = env._action_to_delta(aa)  # expects (d_row, d_col)
                    nr = int(np.clip(row + d_row, 0, env.grid_size - 1))
                    nc = int(np.clip(col + d_col, 0, env.grid_size - 1))
                    ns = state_to_index((nr, nc))

                    # assign reward based on next cell
                    if (nr, nc) == env.goal_pos:
                        r = 10.0
                    elif (nr, nc) in [env.bad1_pos, env.bad2_pos]:
                        r = -10.0
                    else:
                        r = -1.0

                    transitions[s][a].append((p, ns, r))

    # --- Policy Iteration loop ---
    for i in range(max_iterations):
        # Policy Evaluation (iterative until convergence)
        while True:
            delta = 0.0
            for s in range(n_states):
                # keep terminal values fixed
                if s in terminal_states:
                    V[s] = terminal_states[s]
                    continue

                v = V[s]
                new_v = 0.0
                for a, action_prob in enumerate(policy[s]):
                    for (p, ns, r) in transitions[s][a]:
                        # If next state is terminal, do not add gamma * V[ns]
                        if ns in terminal_states:
                            target = r
                        else:
                            target = r + gamma * V[ns]
                        new_v += action_prob * p * target

                V[s] = new_v
                delta = max(delta, abs(v - new_v))

            if delta < theta:
                break

        # Policy Improvement
        policy_stable = True
        for s in range(n_states):
            if s in terminal_states:
                # keep policy arbitrary/deterministic for terminal (won't be used)
                continue

            old_action = int(np.argmax(policy[s]))
            action_values = np.zeros(n_actions)
            for a in range(n_actions):
                for (p, ns, r) in transitions[s][a]:
                    if ns in terminal_states:
                        target = r
                    else:
                        target = r + gamma * V[ns]
                    action_values[a] += p * target

            best_action = int(np.argmax(action_values))
            policy[s] = np.eye(n_actions)[best_action]
            if best_action != old_action:
                policy_stable = False

        print(f"Iteration {i+1}: Policy stable = {policy_stable}")
        if policy_stable:
            break

    return policy, V