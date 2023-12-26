def linear_e_greedy_action_selection(n_actions, epsilon, i, q, t, random_state):
    if t <= n_actions or random_state.rand() < epsilon[i]:
        return random_state.choice(n_actions)
    else:
        argmax = q.argmax()
        return argmax

def exponential_decay_e_greedy_action_selection(n_actions, epsilon, i, q, t, random_state):
    epsilon_start = 1.0
    epsilon_end = 0.001
    steps = len(epsilon)
    delta = (epsilon_end / epsilon_start) ** (1 / steps)

    exploration_threshold = (epsilon[0] * delta ** i)

    if t <= n_actions or random_state.rand() < exploration_threshold:
        return random_state.choice(n_actions)
    else:
        argmax = q.argmax()
        return argmax


def reward_decay_e_greedy_action_selection(n_actions, epsilon, i, q, t, random_state, rewards):
    steps = len(epsilon)

    if t <= n_actions or random_state.rand() < (epsilon[i] ** (rewards + 1)):
        return random_state.choice(n_actions)
    else:
        argmax = q.argmax()
        return argmax
