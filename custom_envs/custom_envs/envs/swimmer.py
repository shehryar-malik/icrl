import numpy as np
from gym.envs.mujoco import swimmer

###############################################################################
# TORQUE CONSTRAINTS
###############################################################################

ACTION_TORQUE_THRESHOLD = 0.5
VIOLATIONS_ALLOWED = 100
class SwimmerTest(swimmer.SwimmerEnv):
    def reset(self):
        ob = super().reset()
        self.current_timestep = 0
        self.violations = 0
        return ob

    def step(self, action):
        next_ob, reward, done, infos = super().step(action)
        # This is to handle the edge case where mujoco_env calls
        # step in __init__ without calling reset with a random
        # action
        try:
            self.current_timestep += 1
            if np.any(np.abs(action) > ACTION_TORQUE_THRESHOLD):
                self.violations += 1
            if self.violations > VIOLATIONS_ALLOWED:
                done = True
                reward = 0
        except:
            pass
        return next_ob, reward, done, infos


##############################################################################
REWARD_TYPE = 'old'         # Which reward to use, traditional or new one?

# =========================================================================== #
#                   Swimmer With Global Postion Coordinates                   #
# =========================================================================== #

class SwimmerWithPos(swimmer.SwimmerEnv):
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
        ])

    def old_reward(self, xposbefore, xposafter, action):
        reward_ctrl = -1e-4 * np.square(action).sum()
        reward_run = abs(xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run

        info = dict(
                reward_run=reward_run,
                reward_ctrl=reward_ctrl,
                xpos=xposafter
                )

        return reward, info


    def new_reward(self, xposbefore, xposafter, action):
        reward_ctrl = -1e-4 * np.square(action).sum()
        reward_dist = abs(xposafter) - abs(xposbefore)
        reward_run  = reward_dist / self.dt

        if np.sign(xposafter) == np.sign(xposbefore):
            reward = reward_ctrl + reward_run
        else:
            reward = 0

        info = dict(
                reward_run=reward_run,
                reward_ctrl=reward_ctrl,
                reward_dist=reward_dist,
                xpos=xposafter
                )

        return reward, info


    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        if REWARD_TYPE == 'new':
            reward, info = self.new_reward(xposbefore,
                                           xposafter,
                                           action)
        elif REWARD_TYPE == 'old':
            reward, info = self.old_reward(xposbefore,
                                           xposafter,
                                           action)
        done = False

        return ob, reward, done, info

class SwimmerWithPosTest(SwimmerWithPos):
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
        ])

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        if REWARD_TYPE == 'new':
            reward, info = self.new_reward(xposbefore,
                                           xposafter,
                                           action)
        elif REWARD_TYPE == 'old':
            reward, info = self.old_reward(xposbefore,
                                           xposafter,
                                           action)
        done = False

        # If agent violates constraint, terminate the episode
        if xposafter <= -3:
            print("Violated constraint in the test environment; terminating episode")
            done = True
            reward = 0

        return ob, reward, done, info


