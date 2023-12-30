import numpy as np


class LinearWrapper:
    def __init__(self, env):
        self.env = env

        self.n_actions = self.env.n_actions
        self.n_states = self.env.n_states
        self.n_features = self.n_actions * self.n_states

    def encode_state(self, s):
        features = np.zeros((self.n_actions, self.n_features))
        for a in range(self.n_actions):
            i = np.ravel_multi_index((s, a), (self.n_states, self.n_actions))
            features[a, i] = 1.0

        return features

    def decode_policy(self, theta):
        policy = np.zeros(self.env.n_states, dtype=int)
        value = np.zeros(self.env.n_states)
        #print('theta:',theta)
        for s in range(self.n_states):
            features = self.encode_state(s)
            q = features.dot(theta)
            #print('s:{} q:{}'.format(s, q))
            policy[s] = np.argmax(q)
            value[s] = np.max(q)

        return policy, value

    def reset(self):
        return self.encode_state(self.env.reset())

    def step(self, action):
        state, reward, done = self.env.step(action)
        return self.encode_state(state), reward, done

    def render(self, policy=None, value=None):
        self.env.render(policy, value)


def e_greedy_action_selection(n_actions, epsilon, i, q, t, random_state):
    if t <= n_actions or random_state.rand() < epsilon[i]:
        return random_state.choice(n_actions)
    else:
        return q.argmax()


def linear_sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    theta = np.zeros(env.n_features)

    for i in range(max_episodes):

        features = env.reset()

        q = features.dot(theta)

        lr = eta[i]

        t = 0

        done = False
        while not done:
            action = e_greedy_action_selection(env.n_actions, epsilon, i, q, t, random_state)

            next_features, reward, done = env.step(action)

            q_next = next_features.dot(theta)

            next_action = e_greedy_action_selection(env.n_actions, epsilon, i, q_next, t + 1, random_state)

            # TD Error
            td_error = reward + gamma * q_next[next_action] - q[action]

            # Update theta. The gradient of q is features!
            theta += lr * td_error * features[action]

            # Move to next action
            features = next_features

            q = q_next

            t += 1

    return theta


def linear_q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    theta = np.zeros(env.n_features)
    for i in range(max_episodes):

        features = env.reset()
        q = features.dot(theta)

        t = 0

        lr = eta[i]
        done = False
        while not done:
            action = e_greedy_action_selection(env.n_actions, epsilon, i, q, t, random_state)

            next_features, reward, done = env.step(action)

            q_next = next_features.dot(theta)

            delta = reward + (gamma * np.max(q_next)) - q[action]

            theta += (lr * delta * features[action])
            q = q_next
            features = next_features
            t += 1

    return theta
