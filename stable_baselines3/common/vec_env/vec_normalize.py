import pickle

import numpy as np

from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper


class VecNormalize(VecEnvWrapper):
    """
    A moving average, normalizing wrapper for vectorized environment.
    has support for saving/loading moving average,

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
        self, venv, training=True, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0, gamma=0.99, epsilon=1e-8
    ):
        VecEnvWrapper.__init__(self, venv)
        self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
        self.ret_rms = RunningMeanStd(shape=())
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        # Returns: discounted rewards
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon
        self.training = training
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward
        self.old_obs = np.array([])
        self.old_reward = np.array([])

    def __getstate__(self):
        """
        Gets state for pickling.

        Excludes self.venv, as in general VecEnv's may not be pickleable."""
        state = self.__dict__.copy()
        # these attributes are not pickleable
        del state["venv"]
        del state["class_attributes"]
        # these attributes depend on the above and so we would prefer not to pickle
        del state["ret"]
        return state

    def __setstate__(self, state):
        """
        Restores pickled state.

        User must call set_venv() after unpickling before using.

        :param state:"""
        self.__dict__.update(state)
        assert "venv" not in state
        self.venv = None

    def set_venv(self, venv):
        """
        Sets the vector environment to wrap to venv.

        Also sets attributes derived from this such as `num_env`.

        :param venv:
        """
        if self.venv is not None:
            raise ValueError("Trying to set venv of already initialized VecNormalize wrapper.")
        VecEnvWrapper.__init__(self, venv)
        if self.obs_rms.mean.shape != self.observation_space.shape:
            raise ValueError("venv is incompatible with current statistics.")
        self.ret = np.zeros(self.num_envs)

    def step_wait(self):
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, news)

        where 'news' is a boolean vector indicating whether each element is new.
        """
        obs, rews, news, infos = self.venv.step_wait()
        self.old_obs = obs
        self.old_reward = rews
        if self.training:
            self.obs_rms.update(obs)
        obs = self.normalize_obs(obs)

        if self.training:
            self._update_reward(rews)
        rews = self.normalize_reward(rews)

        self.ret[news] = 0
        return obs, rews, news, infos

    def _update_reward(self, reward):
        """Update reward normalization statistics."""
        self.ret = self.ret * self.gamma + reward
        self.ret_rms.update(self.ret)

    def normalize_obs(self, obs):
        """
        Normalize observations using this VecNormalize's observations statistics.
        Calling this method does not update statistics.
        """
        if self.norm_obs:
            obs = np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon), -self.clip_obs, self.clip_obs)
        return obs

    def normalize_reward(self, reward):
        """
        Normalize rewards using this VecNormalize's rewards statistics.
        Calling this method does not update statistics.
        """
        if self.norm_reward:
            reward = np.clip(reward / np.sqrt(self.ret_rms.var + self.epsilon), -self.clip_reward, self.clip_reward)
        return reward

    def unnormalize_obs(self, obs):
        if self.norm_obs:
            return (obs * np.sqrt(self.obs_rms.var + self.epsilon)) + self.obs_rms.mean
        return obs

    def unnormalize_reward(self, reward):
        if self.norm_reward:
            return reward * np.sqrt(self.ret_rms.var + self.epsilon)
        return reward

    def get_original_obs(self):
        """
        Returns an unnormalized version of the observations from the most recent
        step or reset.
        """
        return self.old_obs.copy()

    def get_original_reward(self):
        """
        Returns an unnormalized version of the rewards from the most recent step.
        """
        return self.old_reward.copy()

    def reset(self):
        """
        Reset all environments
        """
        obs = self.venv.reset()
        self.old_obs = obs
        self.ret = np.zeros(self.num_envs)
        if self.training:
            self._update_reward(self.ret)
        return self.normalize_obs(obs)

    @staticmethod
    def load(load_path: str, venv: VecEnv) -> "VecNormalize":
        """
        Loads a saved VecNormalize object.

        :param load_path: the path to load from.
        :param venv: the VecEnv to wrap.
        :return:
        """
        with open(load_path, "rb") as file_handler:
            vec_normalize = pickle.load(file_handler)
        vec_normalize.set_venv(venv)
        return vec_normalize

    def save(self, save_path: str) -> None:
        """
        Save current VecNormalize object with
        all running statistics and settings (e.g. clip_obs)

        :param save_path: The path to save to
        """
        with open(save_path, "wb") as file_handler:
            pickle.dump(self, file_handler)


class VecNormalizeWithCost(VecNormalize):
    def __init__(
        self, venv, training=True, norm_obs=True, norm_reward=True, norm_cost=True, cost_info_str='cost',
        clip_obs=10.0, clip_reward=10.0, clip_cost=10.0, reward_gamma=0.99, cost_gamma=0.99, epsilon=1e-8
    ):
        super().__init__(venv=venv, training=training, norm_obs=norm_obs, norm_reward=norm_reward,
                         clip_obs=clip_obs, clip_reward=clip_reward, gamma=reward_gamma, epsilon=epsilon)
        self.norm_cost = norm_cost
        self.cost_str = cost_info_str
        self.clip_cost = clip_cost
        self.gamma = reward_gamma
        self.cost_gamma = cost_gamma
        self.cost_rms = RunningMeanStd(shape=())
        # Returns: discounted rewards
        self.cost_ret = np.zeros(self.num_envs)

    def __getstate__(self):
        """
        Gets state for pickling.
        """
        state = super().__getstate__()
        # these attributes depend on the above and so we would prefer not to pickle
        del state["cost_ret"]
        return state

    def set_venv(self, venv):
        """
        Sets the vector environment to wrap to venv.

        Also sets attributes derived from this such as `num_env`.

        :param venv:
        """
        super().set_venv(venv)
        self.cost_ret = np.zeros(self.num_envs)

    def step_wait(self):
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, news)

        where 'news' is a boolean vector indicating whether each element is new.
        """
        obs, rews, news, infos = super().step_wait()

        # If environment returns desired cost, then handle it
        # We are assuming here that environment always returns cost or does not return cost at all.
        # so will only check the first environment infos.
        if infos[0] is not None and self.cost_str in infos[0].keys():
            cost = [infos[i][self.cost_str] for i in range(len(infos))]
            cost = np.array(cost)
            self.old_cost = cost
            if self.training:
                self._update_cost(cost)
            normalized_cost = self.normalize_cost(cost)
            for i in range(len(infos)):
                infos[i][self.cost_str] = normalized_cost[i]
            self.cost_ret[news] = 0

        return obs, rews, news, infos

    def _update_cost(self, cost):
        """Update reward normalization statistics."""
        self.cost_ret = self.cost_ret * self.cost_gamma + cost
        self.cost_rms.update(self.cost_ret)

    def normalize_cost(self, cost):
        """
        Normalize rewards using this VecNormalize's rewards statistics.
        Calling this method does not update statistics.
        """
        if self.norm_cost:
            cost = np.clip(cost / np.sqrt(self.cost_rms.var + self.epsilon), -self.clip_cost, self.clip_cost)
        return cost

    def unnormalize_cost(self, cost):
        if self.norm_cost:
            return cost * np.sqrt(self.cost_rms.var + self.epsilon)
        return cost

    def get_original_cost(self):
        """
        Returns an unnormalized version of the rewards from the most recent step.
        """
        return self.old_cost.copy()

    def reset(self):
        """
        Reset all environments
        """
        normalized_obs = super().reset()
        self.cost_ret = np.zeros(self.num_envs)
        if self.training:
            self._update_cost(self.cost_ret)
        return normalized_obs
