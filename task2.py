import numpy as np

def policy_evaluation(env, policy, discount_factor, theta, max_iterations):
    # Initialize value array V to zeros
    values = np.zeros(env.n_states)
    for _ in range(max_iterations):
        delta = 0
        for state in range(env.n_states):
            current_value = 0

            # Sum over all actions
            for action, action_prob in enumerate(policy[state]):
                # Sum over all next states and rewards
                for s_prime in range(env.n_states):
                    # Ensure we do not execute an action in the absorbing state
                    if state != env.absorbing_state:
                        prob = env.p(s_prime, state, action)
                        reward = env.r(s_prime, state, action)
                        current_value += action_prob * prob * (reward + discount_factor * values[s_prime])
            
            delta = max(delta, np.abs(current_value - values[state]))
            values[state] = current_value
        
        # Stopping condition
        if delta < theta:
            break
    return values

def policy_improvement(env, value, discount_factor):
    # Initialize policy to zeros
    policy = np.zeros([env.n_states, env.n_actions])

    # Exclude the absorbing state from the loop
    for state in range(env.n_states - 1):
        # Initialize an array for storing the value of each action
        action_values = np.zeros(env.n_actions)

        # Compute the value of each action
        for action in range(env.n_actions):
            for s_prime in range(env.n_states):
                prob = env.p(s_prime, state, action)
                reward = env.r(s_prime, state, action)
                action_values[action] += prob * (reward + discount_factor * value[s_prime])
        
        # Select the best action
        best_action = np.argmax(action_values)

        # Set the probability of selecting the best action to 1
        policy[state, best_action] = 1.0

    # No policy needed for the absorbing state, as the episode ends there
    policy[env.absorbing_state] = 0

    return policy

def policy_iteration(env, gamma, theta, max_iterations, policy=None):
    if policy is None:
        policy = np.zeros((env.n_states, env.n_actions), dtype=np.float64)

        # Initialize a policy with the first action as default
        policy[:, 0] = 1.0
    else:
        policy = np.array(policy, dtype=np.float64)
    
    for _ in range(max_iterations):
        # Calculate the value of the current policy using policy_evaluation
        value = policy_evaluation(env, policy, gamma, theta, max_iterations)

        # Create a new policy with the calculated value using policy_improvement
        new_policy = policy_improvement(env, value, gamma)

        # When the old policy is equal to the new policy, we have reached the optimal policy
        # Break out of the loop
        if np.array_equal(new_policy, policy):
            break

        policy = new_policy
    return policy, value

def value_iteration(env, gamma, theta, max_iterations, value=None):
    value = np.zeros(env.n_states, dtype=np.float64)
    policy = np.zeros((env.n_states, env.n_actions), dtype=np.float64)
    for _ in range(max_iterations):
        delta = 0
        for state in range(env.n_states):
            q_values = np.zeros(env.n_actions)
            for action in range(env.n_actions):
                for next_state in range(env.n_states):
                    reward = env.r(next_state, state, action)
                    q_values[action] += env.p(next_state, state, action) * (reward + gamma * value[next_state])
            best_value = np.max(q_values)
            delta = max(delta, np.abs(best_value - value[state]))
            value[state] = best_value
        if delta < theta:
            break
    for state in range(env.n_states):
        q_values = np.zeros(env.n_actions)
        for action in range(env.n_actions):
            for next_state in range(env.n_states):
                reward = env.r(next_state, state, action)
                q_values[action] += env.p(next_state, state, action) * (reward + gamma * value[next_state])
        best_action = np.argmax(q_values)
        policy[state, best_action] = 1.0
    return policy, value