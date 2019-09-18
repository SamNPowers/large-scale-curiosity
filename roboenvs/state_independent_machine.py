import gym
import numpy as np
from gym.utils.seeding import np_random


class StateIndependentMachine(gym.Env):
    """
    This class outputs a state for a given action that has no dependence on the current state.
    Note that ProcessFrame84 is currently making this greyscale, even though it's generating color.
    """
    # Required by NoopResetEnv. The lack of state means that in some ways all of the actions are no-op actions? I don't want a "retain last observation" action in my set of actions though, so I mostly ignore this.
    _action_meanings = {
        0: "NOOP"
    }

    def __init__(self, num_actions=10, num_states=5):
        # TODO: if I actually want to pass arguments: https://github.com/openai/gym/issues/748
        width = 210
        height = 160
        self.action_space = gym.spaces.Discrete(num_actions)
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                shape=(width, height, 3), dtype=np.uint8)
        self.np_random, seed = gym.utils.seeding.np_random(None)  # Required for AddRandomStateToInfo. TODO: allow seed-setting?

        self._action_state_map = np.random.choice(list(range(num_states)), num_actions)
        self._state_color_map = np.random.randint(256, size=(num_states, 3))

    def get_action_meanings(self):
        return self._action_meanings

    def step(self, action):
        observation = self._generate_observation_for_action(action)
        reward = 0
        done = False
        info = {}
        return observation, reward, done, info

    def reset(self):
        action = np.random.choice(self.action_space.n)
        return self._generate_observation_for_action(action)

    def _generate_observation_for_action(self, action):
        state = self._action_state_map[action]
        color = self._state_color_map[state]

        image = np.zeros(self.observation_space.shape) + color

        return image
