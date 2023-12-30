import unittest
import numpy as np
from environment import FrozenLake
from task2 import policy_evaluation, policy_improvement, policy_iteration, value_iteration

class TestReinforcementLearningMethods(unittest.TestCase):

    def setUp(self):  
        lake = [['&', '.', '.', '.'],
            ['.', '#', '.', '#'],
            ['.', '.', '.', '#'],
            ['#', '.', '.', '$']]
        env = FrozenLake(lake, slip=0.1, max_steps=16, seed=0)

        self.env = env
        self.gamma = 0.9
        self.theta = 0.0001
        self.max_iterations = 4000

    def test_policy_evaluation(self):
        print("Testing Policy Evaluation...")
        random_policy = np.ones([self.env.n_states, self.env.n_actions])
        random_policy[self.env.absorbing_state] = 0
        random_policy /= random_policy.sum(axis=1)[:, None]  # Normalize to get probabilities
        initial_value = np.zeros(self.env.lake.size, dtype=np.float64)
        print(f"Initial Value Function: {initial_value}")
        final_value = policy_evaluation(self.env, random_policy, self.gamma, self.theta, self.max_iterations)
        print(f"Final Value Function: {final_value}")
        self.assertTrue(isinstance(final_value, np.ndarray))
        self.assertEqual(final_value.shape, (self.env.n_states,))

    def test_policy_improvement(self):
        print("Testing Policy Improvement...")
        value = np.zeros(self.env.n_states)
        value[self.env.absorbing_state] = 0
        initial_policy = np.zeros((self.env.n_states, self.env.n_actions), dtype=np.float64)
        print(f"Initial Policy: {initial_policy}")
        improved_policy = policy_improvement(self.env, value, self.gamma)
        print(f"Improved Policy: {improved_policy}")
        self.assertTrue(isinstance(improved_policy, np.ndarray))
        self.assertEqual(improved_policy.shape, (self.env.n_states, self.env.n_actions))

    def test_policy_iteration(self):
        print("Testing Policy Iteration...")
        initial_policy = np.zeros((self.env.n_states, self.env.n_actions), dtype=np.float64)
        print(f"Initial Policy: {initial_policy}")
        final_policy, final_value = policy_iteration(self.env, self.gamma, self.theta, self.max_iterations)
        print(f"Final Policy: {final_policy}")
        print(f"Final Value Function: {final_value}")
        self.assertTrue(isinstance(final_policy, np.ndarray))
        self.assertTrue(isinstance(final_value, np.ndarray))
        self.assertEqual(final_policy.shape, (self.env.n_states, self.env.n_actions))
        self.assertEqual(final_value.shape, (self.env.n_states,))

    def test_value_iteration(self):
        print("Testing Value Iteration...")
        initial_value = np.zeros(self.env.n_states)
        print(f"Initial Value Function: {initial_value}")
        final_policy, final_value = value_iteration(self.env, self.gamma, self.theta, self.max_iterations)
        print(f"Final Policy: {final_policy}")
        print(f"Final Value Function: {final_value}")
        self.assertTrue(isinstance(final_policy, np.ndarray))
        self.assertTrue(isinstance(final_value, np.ndarray))
        self.assertEqual(final_policy.shape, (self.env.n_states, self.env.n_actions))
        self.assertEqual(final_value.shape, (self.env.n_states,))

    @staticmethod
    def main():
        unittest.main()

# This is the typical way to run the tests.
if __name__ == '__main__':
    TestReinforcementLearningMethods.main()
