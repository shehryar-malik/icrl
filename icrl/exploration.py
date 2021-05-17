import gym
import numpy as np
import stable_baselines3.common.callbacks as callbacks
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import create_mlp
from torch.optim import Adam

# =====================================================================================
# EXPLORATION REWARD CALLBACK
# =====================================================================================

class ExplorationRewardCallback(callbacks.BaseCallback):
    """
    (1) Keeps a network which has to predict next_state given current state and action.
    (2) At rollout end error in state prediction is used as reward.
    """
    def __init__(
        self,
        obs_dim,
        acs_dim,
        hidden_layers=[50,50],
        device='cpu',
        verbose: int = 1
    ):
        super(ExplorationRewardCallback, self).__init__(verbose)
        self.obs_dim = obs_dim
        self.acs_dim = acs_dim
        self.predictor_network = nn.Sequential(*create_mlp(obs_dim + acs_dim, obs_dim,
                                                hidden_layers)).to(device)
        self.optimizer = Adam(self.predictor_network.parameters(), lr=3e-3)
        self.loss_fn = nn.MSELoss(reduction='none')

    def _init_callback(self):
        pass

    def _on_step(self):
        pass

    # TODO: Fix them
    def reshape_actions(self, acs):
        if self.is_discrete:
            acs_ = acs.astype(int)
            if len(acs.shape) > 1:
                acs_ = np.squeeze(acs_, axis=-1)
            acs = np.zeros([acs.shape[0], self.acs_dim])
            acs[np.arange(acs_.shape[0]), acs_] = 1.
        return acs

    def _on_rollout_end(self):
        # Get data from buffer
        obs = self.model.rollout_buffer.observations.copy()
        batch_size, n_envs, _ = obs.shape
        obs_t = torch.from_numpy(obs.reshape(-1, self.obs_dim))
        acs_t = torch.from_numpy(self.model.rollout_buffer.actions.copy().reshape(-1, self.acs_dim))
        network_input = torch.cat((obs_t, acs_t), axis=-1)
        predicted_obs = self.predictor_network(network_input)
        target_obs = torch.from_numpy(self.model.rollout_buffer.new_observations.copy().reshape(-1, self.obs_dim))
        loss = self.loss_fn(predicted_obs, target_obs)
        rewards = np.sum(loss.clone().detach().numpy(), axis=-1).reshape(batch_size, n_envs)
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        self.model.rollout_buffer.rewards += rewards

        self.logger.record('exploration/predictor_network_loss', np.mean(rewards))

# =====================================================================================
# Cost Shaping Callback
# =====================================================================================

class CostShapingCallback(callbacks.BaseCallback):
    """
    This callback learns a classifier which outputs 1 for cost states.
    """
    def __init__(
        self,
        true_cost_function,
        obs_dim,
        acs_dim,
        use_nn_for_shaping,
        cost_net_hidden_layers=[50,50],
        device='cpu',
        verbose: int=1,
        ):
        super(CostShapingCallback, self).__init__(verbose)
        self.true_cost_function = true_cost_function
        self.obs_dim = obs_dim
        self.acs_dim = acs_dim
        self.use_nn = use_nn_for_shaping
        self.cost_net_hidden_layers = cost_net_hidden_layers
        self.device = device

    def _initialize_cost_net(self,
                             obs_dim,
                             acs_dim,
                             hidden_layers,
                             device):
        self.cost_net = nn.Sequential(*create_mlp(obs_dim+acs_dim, 1,
                                      hidden_layers), nn.Sigmoid()).to(device)
        #self.cost_loss_fn = nn.MSELoss()
        self.cost_loss_fn = nn.BCELoss()
        self.cost_net_optim = Adam(self.cost_net.parameters(), lr=3e-3)

    def predict_cost(self, observations, actions):
        batch_size, n_envs, obs_dim = observations.shape
        _, _, acs_dim = actions.shape
        observations = observations.reshape(-1, obs_dim)
        actions = actions.reshape(-1, acs_dim)
        obs_and_acs = torch.cat((torch.from_numpy(observations),
                                 torch.from_numpy(actions)),
                                 axis=-1).float()
        predicted_cost = self.cost_net(obs_and_acs)
        return predicted_cost


    def _update_cost_net(self,
                         observations,
                         actions,
                         true_cost):
        predicted_cost = self.predict_cost(observations, actions)
        true_cost = torch.from_numpy(true_cost.reshape(-1, 1)).float()
        cost_loss = self.cost_loss_fn(predicted_cost, true_cost).float()
        self.cost_net_optim.zero_grad()
        cost_loss.backward()
        self.cost_net_optim.step()
        return cost_loss.item()

    def _init_callback(self):
        self._initialize_cost_net(self.obs_dim,
                                  self.acs_dim,
                                  self.cost_net_hidden_layers,
                                  self.device)

    def _on_step(self):
        pass

    def get_true_cost(self, obs, acs):
        return self.true_cost_function(obs, acs)

    def get_shaped_cost(self, obs, acs):
        if self.use_nn:
            return np.log(self.predict_cost(obs, acs).detach().numpy())
        else:
            return np.log(1e-3)*self.get_true_cost(self, obs, acs)

    def _on_rollout_end(self):
        # Get data from buffer
        observations = self.model.rollout_buffer.observations.copy()
        # unormalize observations
        observations = self.training_env.unnormalize_obs(observations)
        actions = self.model.rollout_buffer.actions.copy().astype(float)
        true_costs = self.get_true_cost(observations, actions).astype(float)

        # Update networks
        cost_net_loss = self._update_cost_net(observations, actions, true_costs)

        # Get cost & exploration rewards
        shaped_cost = self.get_shaped_cost(observations, actions).reshape(-1, observations.shape[1])

        self.model.rollout_buffer.rewards += shaped_cost

        # Print relevant data
        self.logger.record('CostShaping/mean_true_cost', np.mean(true_costs))
        self.logger.record('CostShaping/mean_shaped_cost', np.mean(shaped_cost))
        self.logger.record('CostShaping/min_shaped_cost', np.min(shaped_cost))
        self.logger.record('CostShaping/max_shaped_cost', np.max(shaped_cost))
        self.logger.record('CostShaping/cost_network_loss', cost_net_loss)


# =====================================================================================
# Lambda SHAPING CALLBACK
# =====================================================================================

class LambdaShapingCallback(callbacks.BaseCallback):
    """
    This callback trains a neural network to predict cost.
    The neural network is trained using the (observation, action, cost) tuples
    by minimizing the mean squared error between prediction of the net and the
    actual cost and with a regularization constraint which promotes the estimate
    to be smooth.

    In addition to this neural network, this callback weighs cost inversely with
    novelty of the state.

    Returns:
        c_hat = net(obs, acs) * (nu/exploration_reward)
    """
    def __init__(
        self,
        obs_dim,
        acs_dim,
        cost_net_hidden_layers=[50,50],
        predictor_net_hidden_layers=[50,50],
        device='cpu',
        verbose: int = 1
    ):
        super(LambdaShapingCallback, self).__init__(verbose)
        self.obs_dim = obs_dim
        self.acs_dim = acs_dim
        self.cost_net_hidden_layers = cost_net_hidden_layers
        self.predictor_net_hidden_layers = predictor_net_hidden_layers
        self.device = device

    def _initialize_cost_net(self,
                             obs_dim,
                             acs_dim,
                             hidden_layers,
                             regularizer_weight,
                             device):
        self.cost_net = nn.Sequential(*create_mlp(obs_dim+acs_dim, 1,
                                      hidden_layers)).to(device)
        self.cost_loss_fn = nn.MSELoss()
        self.cost_net_optim = Adam(self.cost_net.parameters(), lr=3e-3)

    def _initialize_predictor_net(self,
                                  obs_dim,
                                  acs_dim,
                                  hidden_layers,
                                  device):
        self.predictor_network = nn.Sequential(*create_mlp(obs_dim + acs_dim, obs_dim,
                                                hidden_layers)).to(device)
        self.predictor_net_optim = Adam(self.predictor_network.parameters(), lr=3e-3)
        self.predictor_loss_fn = nn.MSELoss(reduction='none')

    def _update_cost_net(self,
                         observations,
                         actions,
                         true_cost):
        batch_size, n_envs, obs_dim = observations.shape
        _, _, acs_dim = actions.shape
        observations = observations.reshape(-1, obs_dim)
        actions = actions.reshape(-1, acs_dim)
        obs_and_acs = torch.cat((torch.from_numpy(observations),
                                 torch.from_numpy(actions)),
                                 axis=-1)
        predicted_cost = self.cost_net(obs_and_acs)
        true_cost = torch.from_numpy(true_cost.reshape(-1, 1))
        cost_loss = self.cost_loss_fn(predicted_cost, true_cost)
        self.cost_net_optim.zero_grad()
        cost_loss.backward()
        self.cost_net_optim.step()
        return cost_loss.item()

    def _update_predictor_net(self,
                              observations,
                              actions,
                              next_observations):
        batch_size, n_envs, _ = observations.shape
        obs_t = torch.from_numpy(observations.reshape(-1, self.obs_dim))
        acs_t = torch.from_numpy(actions.reshape(-1, self.acs_dim))
        network_input = torch.cat((obs_t, acs_t), axis=-1)
        predicted_obs = self.predictor_network(network_input)
        target_obs = torch.from_numpy(next_observations.reshape(-1, self.obs_dim))
        loss = self.predictor_loss_fn(predicted_obs, target_obs)
        self._exploration_reward = np.sum(loss.clone().detach().numpy(), axis=-1).reshape(batch_size, n_envs)
        self.predictor_net_optim.zero_grad()
        loss.mean().backward()
        self.predictor_net_optim.step()
        return loss.mean().item()

    def _predict_cost(self,
                      observations,
                      actions):
        batch_size, n_envs, obs_dim = observations.shape
        _, _, acs_dim = actions.shape
        observations = observations.reshape(-1, obs_dim)
        actions = actions.reshape(-1, acs_dim)
        obs_and_acs = torch.cat((torch.from_numpy(observation),
                                 torch.from_numpy(actions)),
                                 axis=-1)
        with torch.no_grad():
            predicted_cost = self.cost_net(obs_and_acs)
        return predicted_cost.numpy().reshape(batch_size, n_envs)

    def _init_callback(self):
        self._initialize_cost_net(self.obs_dim,
                                  self.acs_dim,
                                  self.cost_net_hidden_layers,
                                  None, # Not implemented yet
                                  self.device)
        self._initialize_predictor_net(self.obs_dim,
                                  self.acs_dim,
                                  self.predictor_net_hidden_layers,
                                  self.device)

    def _on_step(self):
        pass

    def _on_rollout_end(self):
        # Get data from buffer
        observations = self.model.rollout_buffer.observations.copy()
        next_observations = self.model.rollout_buffer.new_observations.copy()
        actions = self.model.rollout_buffer.actions.copy()
        true_costs = self.model.rollout_buffer.costs.copy()

        # Update networks
        cost_net_loss = self._update_cost_net(observations, actions, true_costs)
        predictor_net_loss = self._update_predictor_net(observations, actions, next_observations)

        # Get cost & exploration rewards
        #cost = self._predict(observations, actions) if self.use_cost_network else true_costs
        exploration_reward = self._exploration_reward  # already computed in predictor net update

        # Weight the cost inversely
        #cost /= (1 + exploration_reward)

        # Set the cost in the buffer
        #self.model.rollout_buffer.costs = costs
        self.model.rollout_buffer.cost_advantages /= (1+exploration_reward)

        # Print relevant data
        self.logger.record('exploration/mean_exploration_reward', np.mean(exploration_reward))
        self.logger.record('exploration/std_exploration_reward', np.std(exploration_reward))
        self.logger.record('exploration/predictor_network_loss', predictor_net_loss)
        self.logger.record('exploration/cost_network_loss', cost_net_loss)



#############################################################################################
# NOVELTY REWARD WRAPPER
# Agent is rewarded higher if it visits states which are different than the ones in buffer.
#############################################################################################

#class NoveltyRewardWrapper(gym.Wrapper):
#    def __init__(self, env):
#        self.obs_buffer = np.zeros((buffer_length, obs_dim))
#        self.current_timestep = 0
#
#    def step(self, action):
#        next_ob, reward, done, info = self.env.step(action)
#        novelty_reward = np.mean(self.queue - next_ob)
#        reward =+= novelty_reward
#        self.obs_buffer[self.current_timestep] = next_ob.copy()
#        self.current_timestep += 1
#        return next_ob, reward, done, info
