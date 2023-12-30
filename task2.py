import numpy as np

def policy_evaluation(env, policy, gamma, theta, max_iterations):
    value = np.zeros(env.lake.size, dtype=np.float64)
    for i in range(max_iterations):
        delta = 0
        for state in range(env.lake.size):
            v = 0
            for action, action_prob in enumerate(policy[state]):
                for next_state in range(env.lake.size):
                    reward = env.r(next_state, state, action)
                    v += action_prob * env.p(next_state, state, action) * (reward + gamma * value[next_state])
            delta = max(delta, np.abs(v - value[state]))
            value[state] = v
        if delta < theta:
            break
    return value

def policy_improvement(env, value, gamma):
    policy = np.zeros((env.lake.size, env.n_actions), dtype=np.float64)
    for state in range(env.lake.size):
        q_values = np.zeros(env.n_actions)
        for action in range(env.n_actions):
            for next_state in range(env.lake.size):
                reward = env.r(next_state, state, action)
                q_values[action] += env.p(next_state, state, action) * (reward + gamma * value[next_state])
        best_action = np.argmax(q_values)
        policy[state, best_action] = 1.0
    return policy

def policy_iteration(env, gamma, theta, max_iterations, policy=None):
    if policy is None:
        policy = np.zeros((env.lake.size, env.n_actions), dtype=np.float64)
        policy[:, 0] = 1.0  # Initialize a policy with the first action as default
    else:
        policy = np.array(policy, dtype=np.float64)
    for i in range(max_iterations):
        value = policy_evaluation(env, policy, gamma, theta, max_iterations)
        new_policy = policy_improvement(env, value, gamma)
        if (new_policy == policy).all():
            break
        policy = new_policy
    return policy, value

def value_iteration(env, gamma, theta, max_iterations, value=None):
    value = np.zeros(env.lake.size, dtype=np.float64)
    policy = np.zeros((env.lake.size, env.n_actions), dtype=np.float64)
    for i in range(max_iterations):
        delta = 0
        for state in range(env.lake.size):
            q_values = np.zeros(env.n_actions)
            for action in range(env.n_actions):
                for next_state in range(env.lake.size):
                    reward = env.r(next_state, state, action)
                    q_values[action] += env.p(next_state, state, action) * (reward + gamma * value[next_state])
            best_value = np.max(q_values)
            delta = max(delta, np.abs(best_value - value[state]))
            value[state] = best_value
        if delta < theta:
            break
    for state in range(env.lake.size):
        q_values = np.zeros(env.n_actions)
        for action in range(env.n_actions):
            for next_state in range(env.lake.size):
                reward = env.r(next_state, state, action)
                q_values[action] += env.p(next_state, state, action) * (reward + gamma * value[next_state])
        best_action = np.argmax(q_values)
        policy[state, best_action] = 1.0
    return policy, value