import argparse
import importlib
import json
import os
import sys
import time

import gym
import numpy as np
import stable_baselines3.common.callbacks as callbacks
from stable_baselines3 import PPOLagrangian
from stable_baselines3.common.vec_env import (VecNormalize,
                                              sync_envs_normalization)

import icrl.utils as utils
import wandb
from icrl.constraint_net import ConstraintNet, plot_constraints
from icrl.gail_utils import GailDiscriminator
from icrl.exploration import ExplorationRewardCallback, LambdaShapingCallback
from icrl.plot_utils import get_plot_func
from icrl.true_constraint_net import get_true_cost_function, null_cost


def cpg(config):
    use_cost_wrapper_train = True
    use_cost_wrapper_eval = True

    # Create the vectorized environments
    train_env = utils.make_train_env(env_id=config.train_env_id,
                                     save_dir=config.save_dir,
                                     use_cost_wrapper=use_cost_wrapper_train,
                                     base_seed=config.seed,
                                     num_threads=config.num_threads,
                                     normalize_obs=not config.dont_normalize_obs,
                                     normalize_reward=not config.dont_normalize_reward,
                                     normalize_cost=not config.dont_normalize_cost,
                                     cost_info_str=config.cost_info_str,
                                     reward_gamma=config.reward_gamma,
                                     cost_gamma=config.cost_gamma)

    eval_env = utils.make_eval_env(env_id=config.eval_env_id,
                                   use_cost_wrapper=use_cost_wrapper_eval,
                                   normalize_obs=not config.dont_normalize_obs)

    is_discrete = isinstance(train_env.action_space, gym.spaces.Discrete)
    obs_dim = train_env.observation_space.shape[0]
    acs_dim = train_env.action_space.n if is_discrete else train_env.action_space.shape[0]

    # Get cost function
    if config.use_null_cost:
        cost_function = null_cost
    elif config.cn_path is None:
        cost_function = get_true_cost_function(config.eval_env_id)
    elif config.load_gail:
        action_low, action_high = None, None
        if isinstance(train_env.action_space, gym.spaces.Box):
            action_low, action_high = train_env.action_space.low, train_env.action_space.high

        gail = GailDiscriminator.load(
                config.cn_path,
                obs_dim=obs_dim,
                acs_dim=acs_dim,
                is_discrete=is_discrete,
                obs_select_dim=config.cn_obs_select_dim,
                acs_select_dim=config.cn_acs_select_dim,
                clip_obs=None,
                obs_mean=None,
                obs_var=None,
                action_low=action_low,
                action_high=action_high,
                device=config.cn_device,
                )

        def cost_function(obs, acs):
            return gail.reward_function(obs, acs, apply_log=False)

        # Assuming we are using 1d or 2d gail here
        gail.plot_HC(os.path.join(config.save_dir, 'gail.png'))

        #plot_constraints(cost_function, eval_env, config.eval_env_id, constraint_net.select_dim,
        #            obs_dim, acs_dim, os.path.join(config.save_dir, 'constraint_net.png'))
    else:
        # Load CN.
        # Set specs
        action_low, action_high = None, None
        if isinstance(train_env.action_space, gym.spaces.Box):
            action_low, action_high = train_env.action_space.low, train_env.action_space.high

        constraint_net = ConstraintNet.load(
                config.cn_path,
                obs_dim=obs_dim,
                acs_dim=acs_dim,
                is_discrete=is_discrete,
                obs_select_dim=config.cn_obs_select_dim,
                acs_select_dim=config.cn_acs_select_dim,
                clip_obs=None,
                obs_mean=None,
                obs_var=None,
                action_low=action_low,
                action_high=action_high,
                device=config.cn_device,
                )
        cost_function = constraint_net.cost_function

        plot_constraints(cost_function, eval_env, config.eval_env_id, constraint_net.select_dim,
                    obs_dim, acs_dim, os.path.join(config.save_dir, 'constraint_net.png'))
    # Pass cost function to environment if applicable
    if use_cost_wrapper_train:
        train_env.set_cost_function(cost_function)
    if use_cost_wrapper_eval:
        eval_env.set_cost_function(cost_function)

    # Define and train model
    policy_kwargs = dict(net_arch=utils.get_net_arch(config))
    if config.policy_name == "ActorTwoCriticsCnnPolicy":
        d = dict(features_dim=config.cnn_features_dim)
        policy_kwargs.update(dict(features_extractor_kwargs=d))

    model = PPOLagrangian(
                policy=config.policy_name,
                env=train_env,
                algo_type='pidlagrangian' if config.use_pid else 'lagrangian',
                learning_rate=config.learning_rate,
                n_steps=config.n_steps,
                batch_size=config.batch_size,
                n_epochs=config.n_epochs,
                reward_gamma=config.reward_gamma,
                reward_gae_lambda=config.reward_gae_lambda,
                cost_gamma=config.cost_gamma,
                cost_gae_lambda=config.cost_gae_lambda,
                clip_range=config.clip_range,
                clip_range_reward_vf=config.clip_range_reward_vf,
                clip_range_cost_vf=config.clip_range_cost_vf,
                ent_coef=config.ent_coef,
                reward_vf_coef=config.reward_vf_coef,
                cost_vf_coef=config.cost_vf_coef,
                max_grad_norm=config.max_grad_norm,
                use_sde=config.use_sde,
                sde_sample_freq=config.sde_sample_freq,
                target_kl=config.target_kl,
                penalty_initial_value=config.penalty_initial_value,
                penalty_learning_rate=config.penalty_learning_rate,
                update_penalty_after=config.update_penalty_after,
                budget=config.budget,
                seed=config.seed,
                device=config.device,
                verbose=config.verbose,
                pid_kwargs=dict(alpha=config.budget,
                                penalty_init=config.penalty_initial_value,
                                Kp=config.proportional_control_coeff,
                                Ki=config.integral_control_coeff,
                                Kd=config.derivative_control_coeff,
                                pid_delay=config.pid_delay,
                                delta_p_ema_alpha=config.proportional_cost_ema_alpha,
                                delta_d_ema_alpha=config.derivative_cost_ema_alpha,),
                policy_kwargs=dict(net_arch=utils.get_net_arch(config))
    )

    # All callbacks
    save_periodically = callbacks.CheckpointCallback(
            config.save_every, os.path.join(config.save_dir, "models"),
            verbose=0
    )
    save_env_stats = utils.SaveEnvStatsCallback(train_env, config.save_dir)
    save_best = callbacks.EvalCallback(
            eval_env, eval_freq=config.eval_every,
            best_model_save_path=config.save_dir, verbose=0,
            deterministic=False,
            callback_on_new_best=save_env_stats
    )
    adjusted_reward = utils.AdjustedRewardCallback(get_true_cost_function(config.eval_env_id))

    # Organize all callbacks in list
    all_callbacks = [save_periodically, save_best, adjusted_reward]

    # Exploration reward callback
    if config.use_curiosity_driven_exploration:
        explorationCallback = ExplorationRewardCallback(obs_dim, acs_dim, device=config.device)
        all_callbacks.append(explorationCallback)

    # Lambda shaping callback
    if config.use_lambda_shaping:
        lambdaCallback = LambdaShapingCallback(obs_dim, acs_dim, device=config.device)
        all_callbacks.append(lambdaCallback)

    # Plotting callback
    if (config.train_env_id in ['C2B-v0', 'PointCircle-v0', 'AntCircle-v0', 'AntWallBroken-v0',
                               'AntWall-v0', 'DD2B-v0'] or 'Point' in config.train_env_id):
        plot_func = get_plot_func(config.train_env_id)
        plot_callback = utils.PlotCallback(
                plot_func, train_env_id=config.train_env_id,
                plot_freq=config.plot_every, plot_save_dir=config.save_dir
        )
        all_callbacks.append(plot_callback)

    # Callback to log actions magnitude
    if any(env in config.train_env_id for env in['Ant', 'HalfCheetah','Point', 'Swimmer', 'Walker', 'HC']):
        all_callbacks.append(utils.LogTorqueCallback())

    # Train
    cost_info_str = config.cost_info_str if config.cost_info_str is not None else cost_function
    model.learn(total_timesteps=int(config.timesteps), cost_function=cost_info_str,
                callback=all_callbacks)

    # Make video of final model
    if not config.wandb_sweep:
        sync_envs_normalization(train_env, eval_env)
        utils.eval_and_make_video(eval_env, model, config.save_dir, "final_policy",
                                  deterministic=False)

    if config.sync_wandb:
        utils.sync_wandb(config.save_dir, 120)

def main():
    start = time.time()
    parser = argparse.ArgumentParser()
    # ======================== Ignore this ========================== #
    parser.add_argument("file_to_run", type=str)
    # ========================== Setup ============================== #
    parser.add_argument("--config_file", "-cf", type=str, default=None)
    parser.add_argument("--project", "-p", type=str, default="ABC")
    parser.add_argument("--name", "-n", type=str, default=None)
    parser.add_argument("--group", "-g", type=str, default=None)
    parser.add_argument("--message", "-m", type=str, default=None)
    parser.add_argument("--device", "-d", type=str, default="cpu")
    parser.add_argument("--verbose", "-v", type=int, default=2)
    parser.add_argument("--wandb_sweep", "-ws", type=bool, default=False)
    parser.add_argument("--sync_wandb", "-sw", action="store_true")
    # ============================ Cost ============================= #
    parser.add_argument("--cost_info_str", "-cis", type=lambda x: None if str(x).lower() == "none" else str(x), default="cost")
    # ======================== Environment ========================== #
    parser.add_argument("--train_env_id", "-tei", type=str, default="HalfCheetah-v3")
    parser.add_argument("--eval_env_id", "-eei", type=str, default="HalfCheetah-v3")
    parser.add_argument("--dont_normalize_obs", "-dno", action="store_true")
    parser.add_argument("--dont_normalize_reward", "-dnr", action="store_true")
    parser.add_argument("--dont_normalize_cost", "-dnc", action="store_true")
    parser.add_argument("--seed", "-s", type=int, default=None)
    # ======================== Networks ============================== #
    parser.add_argument("--policy_name", "-pn", type=str, default="TwoCriticsMlpPolicy")
    parser.add_argument("--shared_layers", "-sl", type=int, default=None, nargs='*')
    parser.add_argument("--policy_layers", "-pl", type=int, default=[64,64], nargs='*')
    parser.add_argument("--reward_vf_layers", "-rl", type=int, default=[64,64], nargs='*')
    parser.add_argument("--cost_vf_layers", "-cl", type=int, default=[64,64], nargs='*')
    parser.add_argument("--cnn_features_dim", "-cfd", type=int, default=512)
    # ========================= Training ============================ #
    parser.add_argument("--timesteps", "-t", type=lambda x: int(float(x)), default=1e6)
    parser.add_argument("--n_steps", "-ns", type=int, default=2048)
    parser.add_argument("--batch_size", "-bs", type=int, default=64)
    parser.add_argument("--n_epochs", "-ne", type=int, default=10)
    parser.add_argument("--num_threads", "-nt", type=int, default=5)
    parser.add_argument("--save_every", "-se", type=float, default=5e5)
    parser.add_argument("--eval_every", "-ee", type=float, default=2048)
    parser.add_argument("--plot_every", "-pe", type=float, default=2048)
    # =========================== MDP =============================== #
    parser.add_argument("--reward_gamma", "-rg", type=float, default=0.99)
    parser.add_argument("--reward_gae_lambda", "-rgl", type=float, default=0.95)
    parser.add_argument("--cost_gamma", "-cg", type=float, default=0.99)
    parser.add_argument("--cost_gae_lambda", "-cgl", type=float, default=0.95)
    # ========================= Losses ============================== #
    parser.add_argument("--clip_range", "-cr", type=float, default=0.2)
    parser.add_argument("--clip_range_reward_vf", "-crv", type=float, default=None)
    parser.add_argument("--clip_range_cost_vf", "-ccv", type=float, default=None)
    parser.add_argument("--ent_coef", "-ec", type=float, default=0.)
    parser.add_argument("--reward_vf_coef", "-rvc", type=float, default=0.5)
    parser.add_argument("--cost_vf_coef", "-cvc", type=float, default=0.5)
    parser.add_argument("--target_kl", "-tk", type=float, default=None)
    parser.add_argument("--max_grad_norm", "-mgn", type=float, default=0.5)
    parser.add_argument("--learning_rate", "-lr", type=float, default=3e-4)
    # ======================= Lagrangian ============================ #
    # (a) General Arguments
    parser.add_argument("--use_pid", "-upid", action="store_true")
    parser.add_argument("--penalty_initial_value", "-piv", type=float, default=1)
    parser.add_argument("--budget", "-b", type=float, default=0.0)
    parser.add_argument("--update_penalty_after", "-upa", type=int, default=1)
    # (b) PID Lagrangian
    parser.add_argument("--proportional_control_coeff", "-kp", type=float, default=10)
    parser.add_argument("--derivative_control_coeff", "-kd", type=float, default=0)
    parser.add_argument("--integral_control_coeff", "-ki", type=float, default=0.0001)
    parser.add_argument("--proportional_cost_ema_alpha", "-pema", type=float, default=0.5)
    parser.add_argument("--derivative_cost_ema_alpha", "-dema", type=float, default=0.5)
    parser.add_argument("--pid_delay", "-pidd", type=int, default=1)
    # (c) Traditional Lagrangian
    parser.add_argument("--penalty_learning_rate", "-plr", type=float, default=0.1,
                        help="Sets Learning Rate of Dual Variables if not using PID Lagrangian.")
    # ======================= Exploration============================= #
    parser.add_argument("--use_sde", "-us", action="store_true")
    parser.add_argument("--use_curiosity_driven_exploration", "-ucde", action="store_true")
    parser.add_argument("--use_lambda_shaping", "-uls", action="store_true")
    parser.add_argument("--sde_sample_freq", "-ssf", type=int, default=-1)
    # ===================== Constraint Net ========================== #
    parser.add_argument("--use_null_cost", "-unc", action="store_true")
    parser.add_argument("--cn_path", "-cp", type=str, default=None)
    parser.add_argument('--cn_obs_select_dim', '-cosd', type=int, default=None, nargs='+')
    parser.add_argument('--cn_acs_select_dim', '-casd', type=int, default=None, nargs='+')
    parser.add_argument('--cn_device', '-cd', type=str, default=None)
    # ============================= GAIL ============================ #
    parser.add_argument("--load_gail", "-lg", action="store_true")
    args = vars(parser.parse_args())

    # Get default config
    default_config, mod_name = {}, ''
    if args["config_file"] is not None:
        if args["config_file"].endswith(".py"):
            mod_name = args["config_file"].replace('/', '.').strip(".py")
            default_config = importlib.import_module(mod_name).config
        elif args["config_file"].endswith(".json"):
            default_config = utils.load_dict_from_json(args["config_file"])
        else:
            raise ValueError("Invalid type of config file")

    # Overwrite config file with parameters supplied through parser
    # Order of priority: supplied through command line > specified in config
    # file > default values in parser
    config = utils.merge_configs(default_config, parser, sys.argv[1:])

    # Choose seed
    if config["seed"] is None:
        config["seed"] = np.random.randint(0,100)

    # Get name by concatenating arguments with non-default values. Default
    # values are either the one specified in config file or in parser (if both
    # are present then the one in config file is prioritized)
    config["name"] = utils.get_name(parser, default_config, config, mod_name)

    # Initialize W&B project
    wandb.init(project=config["project"], name=config["name"], config=config, dir="./cpg",
               group=config["group"])
    wandb.config.save_dir = wandb.run.dir
    config = wandb.config

    print(utils.colorize("Configured folder %s for saving" % config.save_dir,
          color="green", bold=True))
    print(utils.colorize("Name: %s" % config.name, color="green", bold=True))

    # Save config
    utils.save_dict_as_json(config.as_dict(), config.save_dir, "config")

    # Train
    cpg(config)

    end = time.time()
    print(utils.colorize("Time taken: %05.2f minutes" % ((end-start)/60),
          color="green", bold=True))
