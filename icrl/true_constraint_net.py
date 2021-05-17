from functools import partial
import copy
import math

import custom_envs
import custom_envs.envs.utils as ce_utils
import gym
import numpy as np


def get_true_cost_function(env_id):
    """Returns the cost function correpsonding to provided env)"""
    if env_id in ["HCWithPosTest-v0",
                  "WalkerWithPosTest-v0",
                  "SwimmerWithPosTest-v0"]:
        return partial(wall_behind, -3)
    elif env_id == "AntWallTest-v0":
        return partial(wall_behind, -3)
    elif env_id == "AntWallBrokenTest-v0":
        return partial(wall_behind, -3)
    elif env_id in ["PointNullRewardTest-v0", "PointCircleTest-v0", "AntCircleTest-v0"]:
        return partial(wall_behind_and_infront, -3, +3)
    elif env_id == "PointCircleTestBack-v0":
        return partial(wall_behind, -3)
    elif env_id in ["CDD2B-v0", "CC2B-v0", "CDD3B-v0"]:
        dummy_env = gym.make(env_id)
        return partial(bridges, dummy_env)
    elif env_id == "CLGW-v0":
        return lap_grid_world
    elif env_id in ["AntTest-v0", 'HalfCheetahTest-v0', 'Walker2dTest-v0', 'SwimmerTest-v0']:
        return partial(torque_constraint, 0.5)
    else:
        print("Cost function for %s is not implemented yet. Returning null cost function" % env_id)
        return null_cost

# ============================================================================
# General cost functions
# ============================================================================

def wall_behind(pos, obs, acs):
    return (obs[...,0] <= pos)

def wall_infront(pos, obs, acs):
    return (obs[...,0] >= pos)

def wall_behind_and_infront(pos_back, pos_front, obs, acs):
    return (obs[...,0] <= pos_back).astype(np.float32) + (obs[...,0] >= pos_front).astype(np.float32)

def null_cost(x, *args):
    # Zero cost everywhere
    return np.zeros(x.shape[:1])

def torque_constraint(threshold, obs, acs):
    return np.any(np.abs(acs) > threshold, axis=-1)

# ============================================================================
# Specific cost functions
# ============================================================================

def bridges(env, obs, acs):
    obs = copy.deepcopy(obs)
    acs = copy.deepcopy(acs)
    if len(obs.shape) > 2:
        batch_size, n_envs,_ = obs.shape
        cost_shape = (batch_size, n_envs)
        obs = np.reshape(obs, (batch_size*n_envs, -1))
        acs = np.reshape(acs, (batch_size*n_envs,-1))
    else:
        cost_shape = (obs.shape[0])

    cost = []
    for ob, ac in zip(obs, acs):
        # Need to unnormalize obs
        ob = unnormalize(env, ob)

        if 'action_map_dict' in env.__dict__:
            # For discrete classes
            ac = env.action_map_dict[int(ac)]
            next_ob = np.around(ob+ac, 6)
        else:
            # For continuous classes
            ac = np.clip(ac, a_min=env.action_space.low,
                         a_max=env.action_space.high)
            ori = ob[2] + ac[1]
            dx = math.cos(ori) * ac[0]
            dy = math.sin(ori) * ac[0]

            next_ob = ob.copy()
            next_ob[0] = np.clip(ob[0] + dx, -env.size, env.size)
            next_ob[1] = np.clip(ob[1] + dy, -env.size, env.size)

            ob = next_ob[:2]
            next_ob = next_ob[:2]

        if ce_utils.in_regions(ob, next_ob, env.constraint_regions):
            cost += [1]
        else:
            cost += [0]

    cost = np.reshape(np.array(cost), cost_shape)

    return cost

def lap_grid_world(obs, acs):
    cost = []
    for ac in acs:
        if ac == 1:
            cost += [1]
        else:
            cost += [0]
    return np.array(cost)


# ============================================================================
# Utils
# ============================================================================

def unnormalize(env, obs):
    if env.normalize:
        obs += 1
        obs *= (env.observation_space.high - env.observation_space.low)
        obs /= 2
        obs += env.observation_space.low
    return obs
