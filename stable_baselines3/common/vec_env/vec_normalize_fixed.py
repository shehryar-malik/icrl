import pickle

import numpy as np

from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

class FixedMeanStd(RunningMeanStd):
    """
    Provides same api as RunningMeanStd with update method disabled.
    """
    def update(self, *args, **kwargs):
        pass

class VecNormalizeFixed(VecNormalize):
    """
    A normalizing wrapper for vectorized environment using fixed mean and std
    calculated from the given state space limits.

    Basically changes the class of obs_rms in VecNormalize from RunningMeanStd
    to FixedMeanStd.

    :param obs_low: lower limit of observation
    :param obs_high: higher limit of observation
    :param venv: the vectorized environment to wrap
    :param training: Whether to update or not the moving average
    :param norm_obs: Whether to normalize observation or not (default: True)
    :param norm_reward: Whether to normalize rewards or not (default: True)
    :param clip_obs: Max absolute value for observation
    :param clip_reward: Max value absolute for discounted reward
    :param gamma: discount factor
    :param epsilon: To avoid division by zero
    """

    def __init__(
        self, venv, obs_low, obs_high, training=True, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0, gamma=0.99, epsilon=1e-8
    ):
        VecNormalize.__init__(self, venv, training, norm_obs, norm_reward, clip_obs, clip_reward, gamma, epsilon)
        self.obs_rms = FixedMeanStd(shape=self.observation_space.shape)
        self.obs_rms.mean = (obs_low + obs_high)/2
        self.obs_rms.var = ((obs_high - obs_low)/2)**2 - self.epsilon

