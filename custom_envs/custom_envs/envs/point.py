import math
import os

import numpy as np
from gym.envs.mujoco import mujoco_env

ABS_PATH = os.path.abspath(os.path.dirname(__file__))

# Constraint values
X_HIGH = +3
X_LOW  = -3

DEFAULT_CAMERA_CONFIG = {
    'distance': 4.0,
}


# =========================================================================== #
#                           Point With Null Reward                            #
# =========================================================================== #

class PointNullReward(mujoco_env.MujocoEnv):
    def __init__(
            self,
            xml_file=ABS_PATH+"/xmls/point_circle.xml",
            size=40,
            reward_dir=[0.,0.],
            target_dist=10.,
            reward_ctrl_weight=0.0,
            *args,
            **kwargs
        ):

        self.size = size
        self.reward_dir = reward_dir
        self.target_dist = target_dist
        self.reward_ctrl_weight = reward_ctrl_weight
        self._reset_noise_scale = 0.0

        super(PointNullReward, self).__init__(xml_file, 1)

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.flatten(),
            self.data.qvel.flat,
            self.get_body_com("torso").flat,
        ])

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
                low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)
        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation

    def step(self, action):
        action = np.clip(action, -0.25, 0.25)
        qpos = np.copy(self.data.qpos)
        qpos[2] += action[1]
        ori = qpos[2]

        # Compute increment in each direction
        dx = math.cos(ori) * action[0]
        dy = math.sin(ori) * action[0]

        # Ensure that the robot is within reasonable range
        qpos[0] = np.clip(qpos[0] + dx, -self.size, self.size)
        qpos[1] = np.clip(qpos[1] + dy, -self.size, self.size)

        self.set_state(qpos, np.copy(self.data.qvel))
        x, y = qpos[0], qpos[1]

        next_obs = self._get_obs()
        done = False
        reward = +1
        infos = {
                    'action_magnitude': np.sum(np.square(action)),
                    'x_pos': x,
                    'y_pos': y
                }

        return next_obs, reward, done, infos

    def get_xy(self):
        qpos = self.data.qpos
        return qpos[0, 0], qpos[1, 0]

class PointNullRewardTest(PointNullReward):
    def check_for_violation(self, x):
        done = False
        if x > X_HIGH or x < X_LOW:
            done = True
            print("Terminating in True Environment")
        return done

    def step(self, action):
        next_obs, reward, done, infos = super().step(action)
        x = infos['x_pos']
        if done is not True:
            done = self.check_for_violation(x)
        return next_obs, reward, done, infos



# =========================================================================== #
#                           Point With Circle Reward                          #
# =========================================================================== #

class PointCircle(mujoco_env.MujocoEnv):
    def __init__(
            self,
            circle_reward=True,
            start_on_circle=True,
            xml_file=ABS_PATH+"/xmls/point_circle.xml",
            size=40,
            reward_dir=[0.,0.],
            target_dist=10.,
            reward_ctrl_weight=0.0,
            *args,
            **kwargs
        ):

        self.size = size
        self.start_on_circle = start_on_circle
        self.reward_dir = reward_dir
        self.target_dist = target_dist
        self.circle_reward = circle_reward
        self.reward_ctrl_weight = reward_ctrl_weight
        self._reset_noise_scale = 0.0

        super(PointCircle, self).__init__(xml_file, 1)

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.flatten(),
            self.data.qvel.flat,
            self.get_body_com("torso").flat,
        ])

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation

    def new_step(self, action):
        self.do_simulation(action, self.frame_skip)
        qpos = np.copy()

    def step(self, action):
        action = np.clip(action, -0.25, 0.25)
        qpos = np.copy(self.data.qpos)
        qpos[2] += action[1]
        ori = qpos[2]

        # Compute increment in each direction
        dx = math.cos(ori) * action[0]
        dy = math.sin(ori) * action[0]

        # Ensure that the robot is within reasonable range
        qpos[0] = np.clip(qpos[0] + dx, -self.size, self.size)
        qpos[1] = np.clip(qpos[1] + dy, -self.size, self.size)

        self.set_state(qpos, np.copy(self.data.qvel))
        next_obs = self._get_obs()
        reward_ctrl = np.sum(np.square(action))

        x, y = qpos[0], qpos[1]
        reward = y * dx - x * dy
        reward /= (1 + np.abs(np.sqrt(x**2 + y**2) - self.target_dist))

        infos = {'circle_reward': reward,
                 'control_reward': reward_ctrl,
                 'action_1': action[0],
                 'action_2': action[1]}

        return next_obs, reward, False, infos

    def get_xy(self):
        qpos = self.data.qpos
        return qpos[0, 0], qpos[1, 0]

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)


class PointCircleTest(PointCircle):
    def step(self, action):
        action = np.clip(action, -0.25, 0.25)
        qpos = np.copy(self.data.qpos)
        qpos[2] += action[1]
        ori = qpos[2]

        # Compute increment in each direction
        dx = math.cos(ori) * action[0]
        dy = math.sin(ori) * action[0]

        # Ensure that the robot is within reasonable range
        qpos[0] = np.clip(qpos[0] + dx, -self.size, self.size)
        qpos[1] = np.clip(qpos[1] + dy, -self.size, self.size)

        self.set_state(qpos, np.copy(self.data.qvel))
        next_obs = self._get_obs()
        reward_ctrl = np.sum(np.square(action))
        done = False

        x, y = qpos[0], qpos[1]
        reward = y*dx - x*dy
        reward /= (1 + np.abs(np.sqrt(x**2 + y**2) - self.target_dist))
        if x > X_HIGH or x < X_LOW:
            reward = 0
            done = True
            print("Terminating in True Environment")

        infos = {'circle_reward': reward,
                 'control_reward': reward_ctrl,
                 'action_1': action[0],
                 'action_2': action[1]}

        return next_obs, reward, done, infos


class PointCircleTestBack(PointCircle):
    def step(self, action):
        action = np.clip(action, -0.25, 0.25)
        qpos = np.copy(self.data.qpos)
        qpos[2] += action[1]
        ori = qpos[2]

        # Compute increment in each direction
        dx = math.cos(ori) * action[0]
        dy = math.sin(ori) * action[0]

        # Ensure that the robot is within reasonable range
        qpos[0] = np.clip(qpos[0] + dx, -self.size, self.size)
        qpos[1] = np.clip(qpos[1] + dy, -self.size, self.size)

        self.set_state(qpos, np.copy(self.data.qvel))
        next_obs = self._get_obs()
        reward_ctrl = np.sum(np.square(action))
        done = False

        x, y = qpos[0], qpos[1]
        reward = y*dx - x*dy
        reward /= (1 + np.abs(np.sqrt(x**2 + y**2) - self.target_dist))
        if x < X_LOW:
            reward = 0
            done = True
            print("Terminating in True Environment")

        infos = {'circle_reward': reward,
                 'control_reward': reward_ctrl,
                 'action_1': action[0],
                 'action_2': action[1]}

        return next_obs, reward, done, infos



# =========================================================================== #
#                           Point With Reward On Track                        #
# =========================================================================== #

class PointTrack(mujoco_env.MujocoEnv):
    """
    There are two rewards:
        - movement reward: To incentivize movement in anti-clockwise direction.
        - track reward: Agent is given a reward of +50 for being on the track
                        and otherwise.
    """

    def __init__(
            self,
            circle_reward=True,
            xml_file=ABS_PATH+"/xmls/point_track.xml",
            size=40,
            reward_dir=[0.,0.],
            target_dist=10.,
            track_width=2.,
            reward_ctrl_weight=0.,
            reset_noise_scale = 0.,
            *args,
            **kwargs
        ):

        self.size = size
        self.reward_dir = reward_dir
        self.target_dist = target_dist
        self.track_width = track_width
        self.circle_reward = circle_reward
        self.reward_ctrl_weight = reward_ctrl_weight
        self._reset_noise_scale = reset_noise_scale

        super(PointTrack, self).__init__(xml_file, 1)

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.flatten(),
            self.data.qvel.flat,
            self.get_body_com("torso").flat,
        ])

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)
        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation

    def step(self, action):
        action = np.clip(action, -0.25, 0.25)
        qpos = np.copy(self.data.qpos)
        qpos[2] += action[1]
        ori = qpos[2]

        # Compute increment in each direction
        dx = math.cos(ori) * action[0]
        dy = math.sin(ori) * action[0]

        # Ensure that the robot is within reasonable range
        qpos[0] = np.clip(qpos[0] + dx, -self.size, self.size)
        qpos[1] = np.clip(qpos[1] + dy, -self.size, self.size)

        self.set_state(qpos, np.copy(self.data.qvel))
        next_obs = self._get_obs()
        reward_ctrl = np.sum(np.square(action))

        # Movement rewards
        x, y = qpos[0], qpos[1]
        movement_reward = -y * dx + x * dy

        # Track reward
        if np.abs(np.sqrt(x**2 + y**2) - self.target_dist) < self.track_width:
            track_reward = 1
        else:
            track_reward = 0

#        track_reward = (1 if np.abs(np.sqrt(x **2 + y **2) - self.target_dist) < self.track_width
#                         else 1 + np.abs(np.sqrt(x **2 + y **2) - self.target_dist))

        reward = movement_reward + track_reward + self.reward_ctrl_weight*reward_ctrl

        infos = {'movement_reward': movement_reward,
                 'track_reward': track_reward,
                 'control_reward': reward_ctrl,
                 'action_1': action[0],
                 'action_2': action[1]}

        return next_obs, reward, False, infos


class PointCircleTestBack(PointCircle):
    def step(self, action):
        action = np.clip(action, -0.25, 0.25)
        qpos = np.copy(self.data.qpos)
        qpos[2] += action[1]
        ori = qpos[2]

        # Compute increment in each direction
        dx = math.cos(ori) * action[0]
        dy = math.sin(ori) * action[0]

        # Ensure that the robot is within reasonable range
        qpos[0] = np.clip(qpos[0] + dx, -self.size, self.size)
        qpos[1] = np.clip(qpos[1] + dy, -self.size, self.size)

        self.set_state(qpos, np.copy(self.data.qvel))
        next_obs = self._get_obs()
        reward_ctrl = np.sum(np.square(action))
        done = False

        x, y = qpos[0], qpos[1]
        reward = y*dx - x*dy
        reward /= (1 + np.abs(np.sqrt(x**2 + y**2) - self.target_dist))
        if x < X_LOW:
            reward = 0
            done = True
            print("Terminating in True Environment")

        infos = {'circle_reward': reward,
                 'control_reward': reward_ctrl,
                 'action_1': action[0],
                 'action_2': action[1]}

        return next_obs, reward, done, infos


