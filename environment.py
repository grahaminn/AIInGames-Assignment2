import numpy as np
import contextlib

# Configures numpy print options
@contextlib.contextmanager
def _printoptions(*args, **kwargs):
	original = np.get_printoptions()
	np.set_printoptions(*args, **kwargs)
	try:
		yield
	finally: 
		np.set_printoptions(**original)


class EnvironmentModel:
	def __init__(self, n_states, n_actions, seed=None):
		self.n_states = n_states
		self.n_actions = n_actions

		self.random_state = np.random.RandomState(seed)

	def p(self, next_state, state, action):
		raise NotImplementedError()

	def r(self, next_state, state, action):
		raise NotImplementedError()

	def draw(self, state, action):
		p = [self.p(ns, state, action) for ns in range(self.n_states)]
		next_state = self.random_state.choice(self.n_states, p=p)
		reward = self.r(next_state, state, action)

		return next_state, reward


class Environment(EnvironmentModel):
	def __init__(self, n_states, n_actions, max_steps, pi, seed=None):
		EnvironmentModel.__init__(self, n_states, n_actions, seed)

		self.n_steps = 0

		self.max_steps = max_steps
		self.pi = pi
		if self.pi is None:
			self.pi = np.full(n_states, 1./n_states)

		self.state = self.random_state.choice(self.n_states, p=self.pi)

	def reset(self):
		self.n_steps = 0
		self.state = self.random_state.choice(self.n_states, p=self.pi)

		return self.state

	def step(self, action):
		if action < 0 or action >= self.n_actions:
			raise Exception('Invalid action.')

		self.n_steps += 1
		done = (self.n_steps >= self.max_steps)

		self.state, reward = self.draw(self.state, action)

		return self.state, reward, done

	def render(self, policy=None, value=None):
		raise NotImplementedError()

        
class FrozenLake(Environment):
	def __init__(self, lake, slip, max_steps, seed=None):
		"""
		lake: A matrix that represents the lake. For example:
		lake =  [['&', '.', '.', '.'],
			['.', '#', '.', '#'],
			['.', '.', '.', '#'],
			['#', '.', '.', '$']]
		slip: The probability that the agent will slip
		max_steps: The maximum number of time steps in an episode
		seed: A seed to control the random number generator (optional)
		"""
		# start (&), frozen (.), hole (#), goal ($)
		self.lake = np.array(lake)
		self.lake_flat = self.lake.reshape(-1)

		self.slip = slip

		n_states = self.lake.size + 1
		n_actions = 4

		pi = np.zeros(n_states, dtype=float)
		pi[np.where(self.lake_flat == '&')[0]] = 1.0

		self.absorbing_state = n_states - 1

		# TODO:
		Environment.__init__(self, n_states, n_actions, max_steps, pi, seed=seed)

	def step(self, action):
		state, reward, done = Environment.step(self, action)

		done = (state == self.absorbing_state) or done

		return state, reward, done
        
	def p(self, next_state, state, action):
		#w=0,up a=1,left s=2,down d=3,right
		#ok to think of these as hardcoded here in the subclass

		#We can only move to the absorbing state from the absorbing state
		#,or a hole
		in_absorbing_state = state==self.absorbing_state
		if in_absorbing_state or self.lake_flat[state]=='#':
		#	print('in absorbing state')
			return 1.0 if next_state==self.absorbing_state else 0.0

		#If we're at goal, we can only move to ourselves or absorbing state
		at_goal = (self.lake_flat[state]=='$')
		if at_goal:
		#	print('at goal')
			return 1.0 if (next_state==self.absorbing_state) else 0.0

		#now we work out which edges we're next to
		top_edge = state < len(self.lake[0])
		#print('top_edge={}'.format(top_edge))
		bottom_edge = state > ((len(self.lake_flat) - len(self.lake[0])))
		#print('bottom_edge={}'.format(bottom_edge))
		left_edge = (state % len(self.lake[0]) == 0)
		#print('left_edge={}'.format(left_edge))
		right_edge = (state % len(self.lake[0]) == len(self.lake[0])-1)
		#print('right_edge={}'.format(right_edge))
		edge_count = np.sum([top_edge, bottom_edge, left_edge, right_edge])
		#print('edge_count={}'.format(edge_count))
		slip_share = 2 if edge_count==2 and next_state==state else 4

		on_same_line = (next_state - next_state%len(self.lake[0])) == (state - state%len(self.lake[0]))
		#print('on_same_line={}'.format(on_same_line))

		#Where is the next state in relation to the current one?
		next_state_same = state == next_state
		# print('next_state_same={}'.format(next_state_same))
		next_state_above = (state%len(self.lake[0])==next_state%len(self.lake[0])) and next_state < state
		#print('next_state_above={}'.format(next_state_above))
		next_state_below = (state%len(self.lake[0])==next_state%len(self.lake[0])) and next_state > state
		#print('next_state_below={}'.format(next_state_below))
		next_state_left = (next_state == state-1) and on_same_line
		#print('next_state_left={}'.format(next_state_left))
		next_state_right = (next_state == state+1) and on_same_line
		#print('next_state_right={}'.format(next_state_right))

		#Is the next state in the direction of our action?
		next_state_in_direction_of_action = (action==0 and next_state_above) or (action==1 and next_state_left) \
											or (action==2 and next_state_below) or (action==3 and next_state_right)

		#print('next_state_in_direction_of_action={}'.format(next_state_in_direction_of_action))

		#Is the next state next to us?
		next_state_is_adjacent = (next_state == state+1 and on_same_line) or (next_state == state-1 and on_same_line) \
								 or (next_state == state + len(self.lake[0])) \
								 or (next_state == state - len(self.lake[0]))

		#print('next_state_is_adjacent={}'.format(next_state_is_adjacent))

		action_hits_edge = (top_edge and action==0) or (left_edge and action==1) or (bottom_edge and action==2) \
						   or (right_edge and action==3)

		#print('action_hits_edge={}'.format(action_hits_edge))

		successful_with_slip = 1-self.slip+(self.slip / slip_share)
		#print('successful_with_slip={}'.format(successful_with_slip))
		only_slip = self.slip / slip_share
		#print('only_slip={}'.format(only_slip))

		#We either slip to the same state, or bounce back to it from an edge
		if next_state == state:
			if edge_count == 0:
				return 0.0
			else:
				return successful_with_slip if action_hits_edge else only_slip
		#We're trying to get to the state suggested by the action, and it's next to us
		elif next_state_in_direction_of_action and next_state_is_adjacent:
			return successful_with_slip
		#We might slip to one of the other states next to us
		elif next_state_is_adjacent:
			return only_slip
		#Zero probability for anything else
		else:
			return 0.0

	def r(self, next_state, state, action):
		#The agent receives reward 1 upon taking an action at the goal.
		#In every other case, the agent receives zero reward.
		tile = self.lake_flat[state]
		if tile=='$':
			return 1
		else:
			return 0

	def render(self, policy=None, value=None):
		if policy is None:
			lake = np.array(self.lake_flat)

			if self.state < self.absorbing_state:
				lake[self.state] = '@'

				print(lake.reshape(self.lake.shape))
		else:
			# UTF-8 arrows look nicer, but cannot be used in LaTeX
			# https://www.w3schools.com/charsets/ref_utf_arrows.asp
			actions = ['^', '<', '_', '>']

			print('Lake:')
			print(self.lake)

			print('Policy:')
			policy = np.array([actions[a] for a in policy[:-1]])
			print(policy.reshape(self.lake.shape))

			print('Value:')
			with _printoptions(precision=3, suppress=True):
				print(value[:-1].reshape(self.lake.shape))
