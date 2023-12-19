import numpy as np


def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    q = np.zeros((env.n_states, env.n_actions))

    for i in range(max_episodes):
        # initial state for episode i
        state = env.reset()
        #t does not equal i. it increases every time an action is taken in this episode
        t = 0
        # Select action a for state s according to an ϵ-greedy policy based on Q.
        unexplored_actions = range(env.n_actions)
        action = e_greedy_action_selection(env.n_actions, unexplored_actions, epsilon, i, q, t, random_state, state)

        done=False
        while not done:
            t+=1
            next_state, reward, done = env.step(action)
            #print('next_state:{}, reward:{}, done:{}'.format(next_state, reward, done))
            next_action = e_greedy_action_selection(env.n_actions, unexplored_actions, epsilon, i, q, t, random_state, next_state)
            q[state][action] += eta[i]*(reward + gamma * q[next_state][next_action] - q[state][action])
            state = next_state
            action = next_action

    policy = q.argmax(axis=1)
    value = q.max(axis=1)
    return policy, value

def e_greedy_action_selection(n_actions, unexplored_actions, epsilon, i, q, t, random_state, state):
    if t <= n_actions:
        index = random_state.choice(len(unexplored_actions))
        action = unexplored_actions[index]
        np.delete(unexplored_actions, index)
        return action
    elif random_state.rand() < epsilon[i]:
        return random_state.choice(n_actions)
    else:
        return q[state].argmax()


def q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    q = np.zeros((env.n_states, env.n_actions))

    for i in range(max_episodes):
        state = env.reset()

        # t does not equal i. it increases every time an action is taken in this episode
        t=0
        unexplored_actions = range(env.n_actions)

        done = False
        while not done:
            # Choose action using an epsilon-greedy policy
            action = e_greedy_action_selection(env.n_actions, unexplored_actions, epsilon, i, q, t, random_state, state)

            # Take the action and observe the next state and reward
            next_state, reward, done = env.step(action)

            # Q-learning update
            next_action = e_greedy_action_selection(env.n_actions, unexplored_actions, epsilon, i, q, t+1, random_state, next_state)
            td_target = reward + gamma * q[next_state][next_action]
            td_error = td_target - q[state][action]
            q[state][action] += eta[i] * td_error

            # Update state
            state = next_state
            t += 1

    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value
