import numpy as np
from gym.envs.mujoco import walker2d

###############################################################################
# TORQUE CONSTRAINTS
###############################################################################

ACTION_TORQUE_THRESHOLD = 0.5
VIOLATIONS_ALLOWED = 100
class Walker2dTest(walker2d.Walker2dEnv):
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


###############################################################################

REWARD_TYPE = 'old'         # Which reward to use, traditional or new one?

# =========================================================================== #
#                    Walker With Global Postion Coordinates                   #
# =========================================================================== #

class WalkerWithPos(walker2d.Walker2dEnv):
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
        ])

    def old_reward(self, xposbefore, xposafter, action):
        reward_ctrl = -1e-3 * np.square(action).sum()
        reward_run = abs(xposafter - xposbefore) / self.dt
        alive_bonus = 1
        reward = reward_ctrl + reward_run + alive_bonus

        info = dict(
                reward_run=reward_run,
                reward_ctrl=reward_ctrl,
                xpos=xposafter
                )

        return reward, info

    def new_reward(self, xposbefore, xposafter, action):
        reward_ctrl = -1e-3 * np.square(action).sum()
        reward_dist = abs(xposafter)
        reward_run  = reward_dist / self.dt

        reward = reward_dist + reward_ctrl
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
        xposafter, height, ang = self.sim.data.qpos[0:3]
        ob = self._get_obs()

        if REWARD_TYPE == 'new':
            reward, info = self.new_reward(xposbefore,
                                           xposafter,
                                           action)
        elif REWARD_TYPE == 'old':
            reward, info = self.old_reward(xposbefore,
                                           xposafter,
                                           action)

        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        return ob, reward, done, info


class WalkerWithPosTest(WalkerWithPos):
    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter, height, ang = self.sim.data.qpos[0:3]
        ob = self._get_obs()
        reward_ctrl = -1e-3 * np.square(action).sum()
        reward_run = abs(xposafter - xposbefore) / self.dt
        alive_bonus = 1
        if REWARD_TYPE == 'new':
            reward, info = self.new_reward(xposbefore,
                                           xposafter,
                                           action)
        elif REWARD_TYPE == 'old':
            reward, info = self.old_reward(xposbefore,
                                           xposafter,
                                           action)

        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)

        # If agent violates constraint, terminate the episode
        if xposafter <= -3:
            print("Violated constraint in the test environment; terminating episode")
            done = True
            reward = 0

        return ob, reward, done, info
