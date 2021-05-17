import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from gym import spaces
from gym.envs.mujoco import mujoco_env

from collections import namedtuple

from custom_envs.envs.utils import *


LG_LAP_SIZE = 11

class LapGridWorld(mujoco_env.MujocoEnv):
    """
    Constructs a square lap environment with sides having odd length.
    On each side, the middle cell contains a 'coin' collecting which
    gives a reward of +3. This is the 'obsreved reward'.

    The true reward is the number of times am agent does a complete traversal
    of the lap in clockwise fashion.

    Agent always starts at the bottom left of the lap.

    Constraint net is expected to learn to constrain anti-clockwise transitions.
    """
    metadata = {"render.modes": ["rgb_array"]}
    def __init__(
            self,
            max_episode_steps=200,
            reward_scheme='balanced',
            normalize_obs=True
    ):
        """
        Args:
            max_episode_steps (int): Number of maximum steps in environment.
                                     Must always be a multiple (lap_size - 1)*4.
                                     If not then imbalanced reward will not
                                     correspond to correct performance.
            reward_scheme (str): 'balanced' or 'imbalanced'
        """
        all_actions = (0,1)   # [Forward, Backward]
        self.lap_size = (LG_LAP_SIZE//2)*2 + 1
        self.reward_scheme = reward_scheme
        self.max_episode_steps = max_episode_steps
        self.viewer = None
        self.normalize = normalize_obs

        # Define spaces.
        self.observation_space = spaces.Box(
                low=np.array((0,)), high=np.array(((LG_LAP_SIZE-1)*4,)),
                dtype=np.float32)
        self.action_space = spaces.Discrete(2)

        # Initialize
        self.initialize()

    def seed(self, seed):
        # Only for compatibility; environment does not has any randomness
        np.random.seed(seed)

    def initialize(self):
        self.number_of_cells = (self.lap_size - 1)*4
        assert (self.max_episode_steps % self.number_of_cells == 0)
        self.rewards = np.zeros(self.number_of_cells)
        r1 = self.lap_size//2   # first coin
        dist = self.lap_size -1 # distance between coins
        if self.reward_scheme == 'balanced':
            self.rewards[r1] = +3
            self.rewards[r1+dist] = +3
            self.rewards[r1+2*dist] = +3
            self.rewards[r1+3*dist] = +3
        elif self.reward_scheme == 'imbalanced':
            self.rewards[r1] = +1
            self.rewards[r1+dist] = +2
            self.rewards[r1+2*dist] = +3
            self.rewards[r1+3*dist] = +4
        self.start_pos = 0
        self.reset()

    def reset(self):
        self.current_pos = self.start_pos
        self.traversals = 0
        self.current_time = 0
        self.reward_so_far = 0

        return self.normalize_obs(np.array([self.current_pos]))

    def get_next_obs(self, obs, action):
        if action == 0:
            new_position = obs + 1
        elif action == 1:
            new_position = obs - 1
        return new_position

    def step(self, action):
        done = False
        if action == 0:
            self.current_pos += 1
            if self.current_pos == self.number_of_cells:
                self.traversals += 1
                self.current_pos = 0
        elif action == 1:
            self.current_pos -= 1
            if self.current_pos < 0:
                self.current_pos = self.number_of_cells - 1

        self.current_time += 1
        if self.current_time == self.max_episode_steps:
            done = True

        self.reward_so_far += self.rewards[self.current_pos]

        obs = self.normalize_obs(np.array([self.current_pos]))

        return (obs,
                self.rewards[self.current_pos],
                done,
                {"traversals_so_far": self.traversals})

    def _idx_to_xy(self, idx):
        if idx < self.lap_size:
            return int(idx), 0
        elif idx < self.lap_size*2-1:
            return self.lap_size-1, int(idx - self.lap_size + 1)
        elif idx < self.lap_size*3-2:
            return int(self.lap_size*3- 3 -idx), self.lap_size-1
        else:
            return 0, int(self.number_of_cells - idx)


    def render(self, mode=None, camera_id=None):
        agent_position = self.current_pos
        return figure_to_array(self.plot(agent_position))

    def plot(self, agent_position, save_name=None):
        a = np.ones((self.lap_size, self.lap_size))*-1
        b = self.rewards

        a[:,0] = b[:self.lap_size]
        a[-1,1:] = b[self.lap_size:self.lap_size*2-1]
        a[1:,-1] = b[self.lap_size*2-2:self.lap_size*3-3][::-1]
        a[0,1:] = b[self.lap_size*3-3:self.number_of_cells][::-1]

        # Start should be shaded a little lighter
        a[0,0] = -0.4
        a[:,0] = b[:self.lap_size]
        a[-1,1:] = b[self.lap_size:self.lap_size*2-1]
        a[1:,-1] = b[self.lap_size*2-2:self.lap_size*3-3][::-1]
        a[0,1:] = b[self.lap_size*3-3:self.number_of_cells][::-1]

        fig, ax = plt.subplots(1,1,figsize=(15,15))
        c = ax.pcolor(a, edgecolors='w', linewidths=2, cmap='pink_r', vmin=-1.0, vmax=1.0)

        # To detect agent position, add a dummy value to that point
        arr = c.get_array()
        arr[np.ravel_multi_index(self._idx_to_xy(agent_position),
            (self.lap_size, self.lap_size))] += 32

        # Adding text
        for p, value in zip(c.get_paths(), arr):
            x, y = p.vertices[:-2, :].mean(0)
            if value > 31:
                ax.text(x, y, 'A', ha="center", va="center", color='#DE6B1F', fontsize=38)
            #===
            # If you want coins + agent in coins cells, uncomment the following block
            #=== 
            # elif value > 32:
            #     ax.text(0.5*x, 1.05*y, 'A', ha="left", va="top", color='white', fontsize=25)
            #     string = str('\$'*int(value - 32))
            #     print(string)
            #     ax.text(x, 0.95*y, string, ha="center", va="center", color='#FFDF00', fontsize=25)
            elif value > 0:
                string = str('\$'*int(value))
                ax.text(x, y, string, ha="center", va="center", color='#FFDF00', fontsize=25)

        # Add current reward and number of traverals at top
        fig.text(0, 1.04, 'Score: {}/{}'.format(self.reward_so_far,
                                                      self.traversals),
                 fontsize=25, ha='left', va='top', transform=ax.transAxes)
        fig.text(1, 1.04, 'Time: %03d' % self.current_time,
                 fontsize=25, ha='right', va='top', transform=ax.transAxes)

        ob = np.arange(0,40)
        co_ords = []
        for i in ob:
            co_ords.append(self._idx_to_xy(i))
        x, y = zip(*co_ords)
        x, y = np.array(x) + 0.5, np.array(y) + 0.5
        ax.scatter(x, y)

        if save_name is not None:
            fig.savefig(save_name)
        else:
            return fig

    def normalize_obs(self, obs):
        if self.normalize:
            obs = obs-self.observation_space.low
            obs *= 2
            obs /= (self.observation_space.high - self.observation_space.low)
            obs -= 1
        return obs

class ConstrainedLapGridWorld(LapGridWorld):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, action):
        done = False
        if action == 0:
            self.current_pos += 1
            if self.current_pos == self.number_of_cells:
                self.traversals += 1
                self.current_pos = 0
            reward = self.rewards[self.current_pos]
        elif action == 1:
            reward = -1   # penalize the backward action
            done = True

        self.current_time += 1
        if self.current_time == self.max_episode_steps:
            done = True

        self.reward_so_far += reward

        obs = self.normalize_obs(np.array([self.current_pos]))

        return (obs,
                reward,
                done,
                {"traversals_so_far": self.traversals})

    def get_next_obs(self, obs, action):
        if action == 0:
            new_position = obs + 1
        elif action == 1:
            new_position = obs
        return new_position
