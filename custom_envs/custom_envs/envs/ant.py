import numpy as np
import os
from gym.envs.mujoco.ant_v3 import AntEnv

ABS_PATH = os.path.abspath(os.path.dirname(__file__))

###############################################################################
# ANT TORQUE CONSTRAINTS
###############################################################################

ACTION_TORQUE_THRESHOLD = 0.5
VIOLATIONS_ALLOWED = 100
class AntTest(AntEnv):
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
# ANT WALL ENVIRONMENTS
###############################################################################

class AntWall(AntEnv):
    def __init__(
            self,
            healthy_reward=1.0,             # default: 1.0
            terminate_when_unhealthy=False, # default: True
            xml_file=ABS_PATH+"/xmls/ant_circle.xml",
            reset_noise_scale=0.1,
            exclude_current_positions_from_observation=False
    ):
       super(AntWall, self).__init__(
                xml_file=xml_file,
                healthy_reward=healthy_reward,
                terminate_when_unhealthy=terminate_when_unhealthy,
                reset_noise_scale=reset_noise_scale,
                exclude_current_positions_from_observation=exclude_current_positions_from_observation
        )
    def step(self, action):
        xy_position_before = self.get_body_com("torso")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()

        xy_velocity = abs(xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        forward_reward = x_velocity
        healthy_reward = self.healthy_reward

#        rewards = forward_reward + healthy_reward
        distance_from_origin = np.linalg.norm(xy_position_after, ord=2)
        rewards = distance_from_origin + healthy_reward
        costs = ctrl_cost + contact_cost

        reward = rewards - costs
        done = self.done
        observation = self._get_obs()
        info = {
            'reward_forward': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'reward_contact': -contact_cost,
            'reward_survive': healthy_reward,

            'x_position': xy_position_after[0],
            'y_position': xy_position_after[1],
            'distance_from_origin': np.linalg.norm(xy_position_after, ord=2),

            'x_velocity': x_velocity,
            'y_velocity': y_velocity,
            'forward_reward': forward_reward,
        }
        return observation, reward, done, info


class AntWallTest(AntWall):
    def step(self, action):
        observation, reward, done, info = super().step(action)
        #if observation[0] < -3 or observation[0] > 3:
        if observation[0] < -3:
            done = True
            reward = 0
        return observation, reward, done, info


class AntWallBroken(AntWall):
    def step(self, action):
        action[4:] = 0
        return super().step(action)


class AntWallBrokenTest(AntWallTest):
    def step(self, action):
        action[4:] = 0
        return super().step(action)


###############################################################################
# ANT CIRCLE ENVIRONMENTS
###############################################################################


class AntCircle(AntEnv):
    def __init__(
            self,
        # =====================================================================
            target_distance=10,
        # =====================================================================
            xml_file=ABS_PATH+"/xmls/ant_circle.xml",
            ctrl_cost_weight=0.5,
            contact_cost_weight=5e-4,
            healthy_reward=1.0,
            terminate_when_unhealthy=True,
            healthy_z_range=(0.2, 1.0),
            contact_force_range=(-1.0, 1.0),
            reset_noise_scale=0.1,
        # =====================================================================
            exclude_current_positions_from_observation=False
        # =====================================================================
    ):
        # =====================================================================
        self.target_dist = target_distance
        # =====================================================================

        super(AntCircle, self).__init__(
                xml_file=xml_file,
                ctrl_cost_weight=ctrl_cost_weight,
                contact_cost_weight=contact_cost_weight,
                healthy_reward=healthy_reward,
                terminate_when_unhealthy=terminate_when_unhealthy,
                healthy_z_range=healthy_z_range,
                contact_force_range=contact_force_range,
                reset_noise_scale=reset_noise_scale,
                exclude_current_positions_from_observation=exclude_current_positions_from_observation
        )

    def step(self, action):
        xy_position_before = self.get_body_com("torso")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        forward_reward = x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost + contact_cost
        reward = rewards - costs

        # =====================================================================
        x_pos, y_pos = xy_position_after
        reward = -y_pos*x_velocity + x_pos*y_velocity
        reward /= (
                1 + np.abs(np.sqrt(x_pos**2 + y_pos**2)-self.target_dist)
        )
        # =====================================================================

        done = self.done
        observation = self._get_obs()
        info = {
            'reward_forward': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'reward_contact': -contact_cost,
            'reward_survive': healthy_reward,

            'x_position': xy_position_after[0],
            'y_position': xy_position_after[1],
            'distance_from_origin': np.linalg.norm(xy_position_after, ord=2),

            'x_velocity': x_velocity,
            'y_velocity': y_velocity,
            'forward_reward': forward_reward,
        }

        return observation, reward, done, info


class AntCircleTest(AntCircle):
    def step(self, action):
        observation, reward, done, info = super().step(action)
        if observation[0] > 3 or observation[0] < -3:
        #    self.done = True
            done = True
            reward = 0
        return observation, reward, done, info
