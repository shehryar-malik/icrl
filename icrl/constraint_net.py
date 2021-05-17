import os
from itertools import accumulate
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.common.utils import update_learning_rate
from torch import nn
from tqdm import tqdm


class ConstraintNet(nn.Module):
    def __init__(
            self,
            obs_dim: int,
            acs_dim: int,
            hidden_sizes: Tuple[int, ...],
            batch_size: int,
            lr_schedule: Callable[[float], float],
            expert_obs: np.ndarray,
            expert_acs: np.ndarray,
            is_discrete: bool,
            regularizer_coeff: float = 0.,
            obs_select_dim: Optional[Tuple[int, ...]] = None,
            acs_select_dim: Optional[Tuple[int, ...]] = None,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            no_importance_sampling: bool = False,
            per_step_importance_sampling: bool = False,
            clip_obs: Optional[float] = 10.,
            initial_obs_mean: Optional[np.ndarray] = None,
            initial_obs_var: Optional[np.ndarray] = None,
            action_low: Optional[float] = None,
            action_high: Optional[float] = None,
            target_kl_old_new: float = -1,
            target_kl_new_old: float = -1,
            train_gail_lambda: Optional[bool] = False,
            eps: float = 1e-5,
            device: str = "cpu"
        ):
        super(ConstraintNet, self).__init__()

        self.obs_dim = obs_dim
        self.acs_dim = acs_dim
        self.obs_select_dim = obs_select_dim
        self.acs_select_dim = acs_select_dim
        self._define_input_dims()

        self.expert_obs = expert_obs
        self.expert_acs = expert_acs

        self.hidden_sizes = hidden_sizes
        self.batch_size = batch_size
        self.is_discrete = is_discrete
        self.regularizer_coeff = regularizer_coeff
        self.importance_sampling = not no_importance_sampling
        self.per_step_importance_sampling = per_step_importance_sampling
        self.clip_obs = clip_obs
        self.device = device
        self.eps = eps

        self.train_gail_lambda = train_gail_lambda

        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer_class = optimizer_class
        self.lr_schedule = lr_schedule

        self.current_obs_mean = initial_obs_mean
        self.current_obs_var = initial_obs_var
        self.action_low = action_low
        self.action_high = action_high

        self.target_kl_old_new = target_kl_old_new
        self.target_kl_new_old = target_kl_new_old

        self.current_progress_remaining = 1.

        self._build()

    def _define_input_dims(self) -> None:
        self.select_dim = []
        if self.obs_select_dim is None:
            self.select_dim += [i for i in range(self.obs_dim)]
        elif self.obs_select_dim[0] != -1:
            self.select_dim += self.obs_select_dim
        if self.acs_select_dim is None:
            self.select_dim += [i for i in range(self.acs_dim)]
        elif self.acs_select_dim[0] != -1:
            self.select_dim += self.acs_select_dim
        assert len(self.select_dim) > 0, ""

        self.input_dims = len(self.select_dim)

    def _build(self) -> None:

        # Create network and add sigmoid at the end
        self.network = nn.Sequential(
                *create_mlp(self.input_dims, 1, self.hidden_sizes),
                nn.Sigmoid()
        )
        self.network.to(self.device)

        # Build optimizer
        if self.optimizer_class is not None:
            self.optimizer = self.optimizer_class(self.parameters(), lr=self.lr_schedule(1), **self.optimizer_kwargs)
        else:
            self.optimizer = None
        if self.train_gail_lambda:
            self.criterion = nn.BCELoss()

    def forward(self, x: th.tensor) -> th.tensor:
        return self.network(x)

    def cost_function(self, obs: np.ndarray, acs: np.ndarray) -> np.ndarray:
        assert obs.shape[-1] == self.obs_dim, ""
        if not self.is_discrete:
            assert acs.shape[-1] == self.acs_dim, ""

        x = self.prepare_data(obs, acs)
        with th.no_grad():
            out = self.__call__(x)
        cost = 1 - out.detach().cpu().numpy()
        return cost.squeeze(axis=-1)

    def call_forward(self, x: np.ndarray):
        with th.no_grad():
            out = self.__call__(th.tensor(x, dtype=th.float32).to(self.device))
        return out

    def train(
            self,
            iterations: np.ndarray,
            nominal_obs: np.ndarray,
            nominal_acs: np.ndarray,
            episode_lengths: np.ndarray,
            obs_mean: Optional[np.ndarray] = None,
            obs_var: Optional[np.ndarray] = None,
            current_progress_remaining: float = 1,
        ) -> Dict[str, Any]:

        # Update learning rate
        self._update_learning_rate(current_progress_remaining)

        # Update normalization stats
        self.current_obs_mean = obs_mean
        self.current_obs_var = obs_var

        # Prepare data
        nominal_data = self.prepare_data(nominal_obs, nominal_acs)
        expert_data = self.prepare_data(self.expert_obs, self.expert_acs)

        # Save current network predictions if using importance sampling
        if self.importance_sampling:
            with th.no_grad():
                start_preds = self.forward(nominal_data).detach()

        early_stop_itr = iterations
        loss = th.tensor(np.inf)
        for itr in tqdm(range(iterations)):
            # Compute IS weights
            if self.importance_sampling:
                with th.no_grad():
                    current_preds = self.forward(nominal_data).detach()
                is_weights, kl_old_new, kl_new_old = self.compute_is_weights(start_preds.clone(), current_preds.clone(),
                                                                             episode_lengths)
                # Break if kl is very large
                if ((self.target_kl_old_new != -1 and kl_old_new > self.target_kl_old_new) or
                    (self.target_kl_new_old != -1 and kl_new_old > self.target_kl_new_old)):
                    early_stop_itr = itr
                    break
            else:
                is_weights = th.ones(nominal_data.shape[0])

            # Do a complete pass on data
            for nom_batch_indices, exp_batch_indices in self.get(nominal_data.shape[0], expert_data.shape[0]):
                # Get batch data
                nominal_batch = nominal_data[nom_batch_indices]
                expert_batch = expert_data[exp_batch_indices]
                is_batch = is_weights[nom_batch_indices][...,None]

                # Make predictions
                nominal_preds = self.__call__(nominal_batch)
                expert_preds = self.__call__(expert_batch)

                # Calculate loss
                if self.train_gail_lambda:
                    nominal_loss = self.criterion(nominal_preds, th.zeros(*nominal_preds.size()))
                    expert_loss = self.criterion(expert_preds, th.ones(*expert_preds.size()))
                    regularizer_loss = th.tensor(0)
                    loss = nominal_loss + expert_loss
                else:
                    expert_loss = th.mean(th.log(expert_preds + self.eps))
                    nominal_loss = th.mean(is_batch * th.log(nominal_preds + self.eps))
                    regularizer_loss = self.regularizer_coeff * (th.mean(1-expert_preds) + th.mean(1-nominal_preds))
                    loss = (-expert_loss + nominal_loss) + regularizer_loss

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        bw_metrics =  {"backward/cn_loss": loss.item(),
                       "backward/expert_loss": expert_loss.item(),
                       "backward/unweighted_nominal_loss": th.mean(th.log(nominal_preds + self.eps)).item(),
                       "backward/nominal_loss": nominal_loss.item(),
                       "backward/regularizer_loss": regularizer_loss.item(),
                       "backward/is_mean": th.mean(is_weights).detach().item(),
                       "backward/is_max": th.max(is_weights).detach().item(),
                       "backward/is_min": th.min(is_weights).detach().item(),
                       "backward/nominal_preds_max": th.max(nominal_preds).item(),
                       "backward/nominal_preds_min": th.min(nominal_preds).item(),
                       "backward/nominal_preds_mean": th.mean(nominal_preds).item(),
                       "backward/expert_preds_max": th.max(expert_preds).item(),
                       "backward/expert_preds_min": th.min(expert_preds).item(),
                       "backward/expert_preds_mean": th.mean(expert_preds).item(),}
        if self.importance_sampling:
            stop_metrics = {"backward/kl_old_new": kl_old_new.item(),
                            "backward/kl_new_old": kl_new_old.item(),
                            "backward/early_stop_itr": early_stop_itr}
            bw_metrics.update(stop_metrics)

        return bw_metrics

    def compute_is_weights(self, preds_old: th.Tensor, preds_new: th.Tensor, episode_lengths: np.ndarray) -> th.tensor:
        with th.no_grad():
            n_episodes = len(episode_lengths)
            cumulative = [0] + list(accumulate(episode_lengths))

            ratio = (preds_new + self.eps) / (preds_old + self.eps)
            prod = [th.prod(ratio[cumulative[j]:cumulative[j+1]])
                    for j in range(n_episodes)]
            prod = th.tensor(prod)
            normed = n_episodes * prod / (th.sum(prod) + self.eps)

            if self.per_step_importance_sampling:
                is_weights = th.tensor(ratio/th.mean(ratio))
            else:
                is_weights = []
                for length, weight in zip(episode_lengths, normed):
                    is_weights += [weight] * length
                is_weights = th.tensor(is_weights)

            # Compute KL(old, current)
            kl_old_new = th.mean(-th.log(prod+self.eps))
            # Compute KL(current, old)
            prod_mean = th.mean(prod)
            kl_new_old = th.mean((prod-prod_mean)*th.log(prod+self.eps)/(prod_mean+self.eps))

        return is_weights.to(self.device), kl_old_new, kl_new_old

    def prepare_data(
            self,
            obs: np.ndarray,
            acs: np.ndarray,
    ) -> th.tensor:

        obs = self.normalize_obs(obs, self.current_obs_mean, self.current_obs_var, self.clip_obs)
        acs = self.reshape_actions(acs)
        acs = self.clip_actions(acs, self.action_low, self.action_high)

        concat = self.select_appropriate_dims(np.concatenate([obs,acs], axis=-1))

        return th.tensor(concat, dtype=th.float32).to(self.device)

    def select_appropriate_dims(self, x: Union[np.ndarray, th.tensor]) -> Union[np.ndarray, th.tensor]:
        return x[...,self.select_dim]

    def normalize_obs(self, obs: np.ndarray, mean: Optional[float] = None, var: Optional[float] = None,
                      clip_obs: Optional[float] = None) -> np.ndarray:
        if mean is not None and var is not None:
            mean, var = mean[None], var[None]
            obs = (obs - mean) / np.sqrt(var + self.eps)
        if clip_obs is not None:
            obs = np.clip(obs, -self.clip_obs, self.clip_obs)

        return obs

    def reshape_actions(self, acs):
        if self.is_discrete:
            acs_ = acs.astype(int)
            if len(acs.shape) > 1:
                acs_ = np.squeeze(acs_, axis=-1)
            acs = np.zeros([acs.shape[0], self.acs_dim])
            acs[np.arange(acs_.shape[0]), acs_] = 1.

        return acs

    def clip_actions(self, acs: np.ndarray, low: Optional[float] = None, high: Optional[float] = None) -> np.ndarray:
        if high is not None and low is not None:
            acs = np.clip(acs, low, high)

        return acs

    def get(self, nom_size: int, exp_size: int) -> np.ndarray:
        if self.batch_size is None:
            yield np.arange(nom_size), np.arange(exp_size)
        else:
            size = min(nom_size, exp_size)
            indices = np.random.permutation(size)

            batch_size = self.batch_size
            # Return everything, don't create minibatches
            if batch_size is None:
                batch_size = size

            start_idx = 0
            while start_idx < size:
                batch_indices = indices[start_idx:start_idx+batch_size]
                yield batch_indices, batch_indices
                start_idx += batch_size

    def _update_learning_rate(self, current_progress_remaining) -> None:
        self.current_progress_remaining = current_progress_remaining
        update_learning_rate(self.optimizer, self.lr_schedule(current_progress_remaining))

    def save(self, save_path):
        state_dict = dict(
                cn_network=self.network.state_dict(),
                cn_optimizer=self.optimizer.state_dict(),
                obs_dim=self.obs_dim,
                acs_dim=self.acs_dim,
                is_discrete=self.is_discrete,
                obs_select_dim=self.obs_select_dim,
                acs_select_dim=self.acs_select_dim,
                clip_obs=self.clip_obs,
                obs_mean=self.current_obs_mean,
                obs_var=self.current_obs_var,
                action_low=self.action_low,
                action_high=self.action_high,
                device=self.device,
                hidden_sizes=self.hidden_sizes
        )
        th.save(state_dict, save_path)

    def _load(self, load_path):
        state_dict = th.load(load_path)
        if "cn_network" in state_dict:
            self.network.load_state_dict(dic["cn_network"])
        if "cn_optimizer" in state_dict and self.optimizer is not None:
            self.optimizer.load_state_dict(dic["cn_optimizer"])

    # Provide basic functionality to load this class
    @classmethod
    def load(
            cls,
            load_path: str,
            obs_dim: Optional[int] = None,
            acs_dim: Optional[int] = None,
            is_discrete: bool = None,
            obs_select_dim: Optional[Tuple[int, ...]] = None,
            acs_select_dim: Optional[Tuple[int, ...]] = None,
            clip_obs: Optional[float] = None,
            obs_mean: Optional[np.ndarray] = None,
            obs_var: Optional[np.ndarray] = None,
            action_low: Optional[float] = None,
            action_high: Optional[float] = None,
            device: str = "auto"
        ):

        state_dict = th.load(load_path)
        # If value isn't specified, then get from state_dict
        if obs_dim is None:
            obs_dim = state_dict["obs_dim"]
        if acs_dim is None:
            acs_dim = state_dict["acs_dim"]
        if is_discrete is None:
            is_discrete = state_dict["is_discrete"]
        if obs_select_dim is None:
            obs_select_dim = state_dict["obs_select_dim"]
        if acs_select_dim is None:
            acs_select_dim = state_dict["acs_select_dim"]
        if clip_obs is None:
            clip_obs = state_dict["clip_obs"]
        if obs_mean is None:
            obs_mean = state_dict["obs_mean"]
        if obs_var is None:
            obs_var = state_dict["obs_var"]
        if action_low is None:
            action_low = state_dict["action_low"]
        if action_high is None:
            action_high = state_dict["action_high"]
        if device is None:
            device = state_dict["device"]

        # Create network
        hidden_sizes = state_dict["hidden_sizes"]
        constraint_net = cls(
                obs_dim, acs_dim, hidden_sizes, None, None, None, None,
                is_discrete, None, obs_select_dim, acs_select_dim, None,
                None, None, clip_obs, obs_mean, obs_var, action_low, action_high,
                None, None, device
        )
        constraint_net.network.load_state_dict(state_dict["cn_network"])

        return constraint_net


# =====================================================================
# Plotting utilities
# =====================================================================

# The following functions exploit knowledge about the environment

def plot_constraints(cost_function, env, env_id, select_dim, obs_dim, acs_dim,
                     save_name, observations=None):
    if env_id in ['PointEnv-v0',
                  'PointEnvTest-v0',
                  'HCWithPos-v0',
                  'AntWallTest-v0',
                  'HCWithPosTest-v0',
                  'PointCircle-v0',
                  'PointCircleTest-v0',
                  'WalkerWithPos-v0',
                  'WalkerWithPosTest-v0']:
        plot_for_gym_envs(cost_function, env, env_id, select_dim, obs_dim, acs_dim,
                          save_name, observations)
    elif env_id in ["D2B-v0", "DD2B-v0", "CDD2B-v0", "TCDD2B-v0",
                    "D3B-v0", "DD3B-v0", "CDD3B-v0", "TCDD3B-v0"]:
        plot_for_bridges_envs(cost_function, save_name, observations)
    elif env_id in ["LGW-v0", "CLGW-v0"]:
        pass
        #plot_for_lap_env(cost_function, save_name, observations)
    else:
        print("Env id not recognized; skipping plotting of constraint net")

def plot_for_gym_envs(cost_function, env, env_id, select_dim, obs_dim, acs_dim,
                     save_name, obs=None):
    if len(select_dim) > 2:
        print("Cannot plot; constraint net has more than two dimensions.")
        return

    x_range = [-12, 12] if "Point" in env_id else [-20,20]

    if len(select_dim) == 1:
        fig, ax = plt.subplots(1,2,figsize=(30,15))
        num_points = 1000
        obs_all = np.linspace(x_range[0], x_range[1], num_points)[...,None]
        obs_all = np.concatenate((obs_all, np.zeros((num_points,obs_dim-1))), axis=-1)

        action = np.zeros((num_points, acs_dim))
        preds = 1-cost_function(obs_all, action)
        ax[0].plot(obs_all, preds)
        if obs is not None:
            ax[0].scatter(obs[...,0], 0.2 + np.zeros(obs.shape[0]))
            ax[1].hist(obs[:,0], bins=40, range=(-20, 20))
            ax[1].set_axisbelow(True)
            # Turn on the minor TICKS, which are required for the minor GRID
            ax[1].minorticks_on()
            ax[1].grid(which='major', linestyle='-', linewidth='0.5', color='red')
            ax[1].grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        ax[0].set_ylim([0,1])
        ax[0].set_xlim(x_range)
        ax[0].set_axisbelow(True)
        # Turn on the minor TICKS, which are required for the minor GRID
        ax[0].minorticks_on()
        ax[0].grid(which='major', linestyle='-', linewidth='0.5', color='red')
        ax[0].grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        fig.savefig(save_name)
        plt.close(fig=fig)

    elif len(select_dim) == 2:
        fig, ax = plt.subplots(1,1,figsize=(30,15))
        r = np.arange(-20,20,0.1)
        X, Y = np.meshgrid(r, r)
        obs_all = np.concatenate([X.reshape([-1,1]), Y.reshape([-1,1])], axis=-1)
        obs_all = np.concatenate((obs_all, np.zeros((np.size(X),obs_dim-2))), axis=-1)

        action = np.zeros((np.size(X), acs_dim))
        outs = 1-cost_function(obs_all, action)
        im = ax.imshow(outs.reshape(X.shape), extent=[-20,20,-20,20], cmap='jet_r',
                                vmin=0, vmax=1, origin='lower')
        #axs[action].set_title('Action: %s' % action_desc[action])
        fig.colorbar(im, ax=ax)

        if obs is not None:
            obs = np.clip(obs, -20, 20)
            ax.scatter(obs[...,0], obs[...,1], clip_on=False)
        ax.set_ylim([-20, 20])
        ax.set_xlim([-20, 20])
        plt.grid('on')
        fig.savefig(save_name)
        plt.close(fig=fig)

def plot_for_bridges_envs(cost_function, save_name, observations=None):
    if observations is not None:
        # Unnormalize
        observations = 10*(observations+1)
    action_desc = ['right', 'left', 'up', 'bottom']
    r = np.arange(0,20,0.1)
    X, Y = np.meshgrid(r, r)
    obs = np.concatenate([X.reshape([-1,1]), Y.reshape([-1,1])], axis=-1)
    # Normalize
    obs = obs/10 - 1

    fig, axs = plt.subplots(2,2)
    axs = [ax for axs_ in axs for ax in axs_]
    fig.set_size_inches(20, 20)
    for action in range(4):
        acs = action*np.ones([obs.shape[0],1])
        outs = 1-cost_function(obs, acs)
        im = axs[action].imshow(outs.reshape(X.shape), extent=[0,20,0,20], cmap='jet_r',
                                vmin=0, vmax=1, origin='lower')
        axs[action].set_title('Action: %s' % action_desc[action])
        fig.colorbar(im, ax=axs[action])

        if observations is not None:
            axs[action].scatter(observations[:,0], observations[:,1], clip_on=False)

        axs[action].set_xlim(0, 20)
        axs[action].set_ylim(0, 20)

    fig.savefig(save_name)
    plt.close(fig=fig)

def plot_for_lap_env(cost_function, save_name, obs=None):
    # Assumes a lap size of 11
    LAP_SIZE = 11
    action_desc = ['Forward', 'Backward']
    number_of_cells = (LAP_SIZE-1)*4
    all_obs = np.arange(number_of_cells)[:,None]
    fig, ax = plt.subplots(1,2,figsize=(30,15))
    if obs is not None: obs = np.round((obs[:,:1] + 1)*(number_of_cells/2))
    for action in range(2):
        ac = action*np.ones([all_obs.shape[0], 1])
        outs = 1-cost_function(all_obs, ac)
        outs = _reshape_lap_to_grid(outs)
        # Plotting
        c = ax[action].pcolor(outs, edgecolors='w', linewidths=2, cmap='jet_r', vmin=-0.0, vmax=1.0)
        ax[action].set_title('Action: %s' % action_desc[action])
        if obs is not None:
            co_ords = []
            for i in range(obs.shape[0]):
                for j in range(obs.shape[1]):
                    co_ords.append(_idx_to_xy(obs[i,j]))
            x, y = zip(*co_ords)
            x, y = np.array(x) + 0.5, np.array(y) + 0.5
            ax[action].scatter(x, y)

    plt.axis('off')
    fig.savefig(save_name)
    plt.close(fig=fig)


if __name__=='__main__':
    from icrl.true_constraint_net import get_true_cost_function
    cn = get_true_cost_function('CDD2B-v0')
    plot_for_bridges_envs(cn, "eev.png", None)
