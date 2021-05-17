import argparse
import importlib
import json
import os
import pickle
import sys
import time

import gym
import numpy as np
import stable_baselines3.common.callbacks as callbacks
from stable_baselines3 import PPO, PPOLagrangian
from stable_baselines3.common import logger
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import (VecNormalize,
                                              sync_envs_normalization)

import icrl.utils as utils
import wandb
from icrl.true_constraint_net import get_true_cost_function
from icrl.gail_utils import GailCallback, GailDiscriminator
from icrl.plot_utils import plot_obs_point, get_plot_func
from icrl.exploration import CostShapingCallback


def load_expert_data(expert_path, num_rollouts):
    expert_mean_reward = []
    for i in range(num_rollouts):
        with open(os.path.join(expert_path, "files/EXPERT/rollouts", "%s.pkl"%str(i)), "rb") as f:
            data = pickle.load(f)

        if i == 0:
            expert_obs = data['observations']
            expert_acs = data['actions']
        else:
            expert_obs = np.concatenate([expert_obs, data['observations']], axis=0)
            expert_acs = np.concatenate([expert_acs, data['actions']], axis=0)

        expert_mean_reward.append(data['rewards'])

    expert_mean_reward = np.mean(expert_mean_reward)
    expert_mean_length = expert_obs.shape[0]/num_rollouts

    return (expert_obs, expert_acs), expert_mean_reward


def gail(config):
    # Create the vectorized environments
    train_env = utils.make_train_env(env_id=config.train_env_id,
                                     save_dir=config.save_dir,
                                     use_cost_wrapper=False,
                                     base_seed=config.seed,
                                     num_threads=config.num_threads,
                                     normalize_obs=not config.dont_normalize_obs,
                                     normalize_reward=not config.dont_normalize_reward,
                                     normalize_cost=False,
                                     reward_gamma=config.reward_gamma
                                     )

    # We don't need cost when taking samples
    sampling_env = utils.make_eval_env(env_id=config.train_env_id,
                                       use_cost_wrapper=False,
                                       normalize_obs=not config.dont_normalize_obs)

    eval_env = utils.make_eval_env(env_id=config.eval_env_id,
                                   use_cost_wrapper=False,
                                   normalize_obs=not config.dont_normalize_obs)

    # Set specs
    is_discrete = isinstance(train_env.action_space, gym.spaces.Discrete)
    obs_dim = train_env.observation_space.shape[0]
    acs_dim = train_env.action_space.n if is_discrete else train_env.action_space.shape[0]

    action_low, action_high = None, None
    if isinstance(train_env.action_space, gym.spaces.Box):
        action_low, action_high = train_env.action_space.low, train_env.action_space.high

    # Load expert data
    (expert_obs, expert_acs), expert_mean_reward = load_expert_data(config.expert_path, config.expert_rollouts)
    expert_agent = PPOLagrangian.load(os.path.join(config.expert_path, "files/best_model.zip"))

    # Logger
    gail_logger = logger.HumanOutputFormat(sys.stdout)

    # Do we want to restore gail from a saved model?
    if config.gail_path is not None:
        discriminator = GailDiscriminator.load(
                        config.gail_path,
                        obs_dim=obs_dim,
                        acs_dim=acs_dim,
                        is_discrete=is_discrete,
                        expert_obs=expert_obs,
                        expert_acs=expert_acs,
                        obs_select_dim=config.disc_obs_select_dim,
                        acs_select_dim=config.disc_acs_select_dim,
                        clip_obs=None,
                        obs_mean=None,
                        obs_var=None,
                        action_low=action_low,
                        action_high=action_high,
                        device=config.device,
                        )
        discriminator.freeze_weights = config.freeze_gail_weights
    else:    # Initialize GAIL and setup its callback
        discriminator = GailDiscriminator(
                obs_dim,
                acs_dim,
                config.disc_layers,
                config.disc_batch_size,
                get_schedule_fn(config.disc_learning_rate),
                expert_obs,
                expert_acs,
                is_discrete,
                config.disc_obs_select_dim,
                config.disc_acs_select_dim,
                clip_obs=config.clip_obs,
                initial_obs_mean=None,
                initial_obs_var=None,
                action_low=action_low,
                action_high=action_high,
                num_spurious_features=config.num_spurious_features,
                freeze_weights=config.freeze_gail_weights,
                eps=config.disc_eps,
                device=config.device
                )

    true_cost_function = get_true_cost_function(config.eval_env_id)

    if config.use_cost_shaping_callback:
        costShapingCallback = CostShapingCallback(true_cost_function,
                                                  obs_dim,
                                                  acs_dim,
                                                  use_nn_for_shaping=config.use_cost_net)
        all_callbacks = [costShapingCallback]
    else:
        plot_disc = True if config.train_env_id in ['DD2B-v0', 'DD3B-v0', 'CDD2B-v0', 'CDD3B-v0'] else False
        if config.disc_obs_select_dim is not None and config.disc_acs_select_dim is not None:
            plot_disc = True if (len(config.disc_obs_select_dim) < 3
                            and config.disc_acs_select_dim[0] == -1) else False
        gail_update = GailCallback(discriminator, config.learn_cost, true_cost_function,
                                   config.save_dir, plot_disc=plot_disc)
        all_callbacks = [gail_update]


    # Define and train model
    model = PPO(
                policy=config.policy_name,
                env=train_env,
                learning_rate=config.learning_rate,
                n_steps=config.n_steps,
                batch_size=config.batch_size,
                n_epochs=config.n_epochs,
                gamma=config.reward_gamma,
                gae_lambda=config.reward_gae_lambda,
                clip_range=config.clip_range,
                clip_range_vf=config.clip_range_reward_vf,
                ent_coef=config.ent_coef,
                vf_coef=config.reward_vf_coef,
                max_grad_norm=config.max_grad_norm,
                use_sde=config.use_sde,
                sde_sample_freq=config.sde_sample_freq,
                target_kl=config.target_kl,
                seed=config.seed,
                device=config.device,
                verbose=config.verbose,
                policy_kwargs=dict(net_arch=utils.get_net_arch(config))
    )

    # All callbacks
    save_periodically = callbacks.CheckpointCallback(
            config.save_every, os.path.join(config.save_dir, "models"),
            verbose=0
    )
    save_env_stats = utils.SaveEnvStatsCallback(train_env, config.save_dir)
    save_best = callbacks.EvalCallback(
            eval_env, eval_freq=config.eval_every, deterministic=False,
            best_model_save_path=config.save_dir, verbose=0,
            callback_on_new_best=save_env_stats
    )
    plot_func = get_plot_func(config.train_env_id)
    plot_callback = utils.PlotCallback(
            plot_func, train_env_id=config.train_env_id,
            plot_freq=config.plot_every, plot_save_dir=config.save_dir
    )

    # Organize all callbacks in list
    all_callbacks.extend([save_periodically, save_best, plot_callback])

    # Train
    model.learn(total_timesteps=int(config.timesteps),
                callback=all_callbacks)

    # Save final discriminator
    if not config.freeze_gail_weights:
        discriminator.save(os.path.join(config.save_dir, "gail_discriminator.pt"))

    # Save normalization stats
    if isinstance(train_env, VecNormalize):
        train_env.save(os.path.join(config.save_dir, "train_env_stats.pkl"))

    # Make video of final model
    if not config.wandb_sweep:
        sync_envs_normalization(train_env, eval_env)
        utils.eval_and_make_video(eval_env, model, config.save_dir, "final_policy")

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
    parser.add_argument("--group", "-g", type=str, default=None)
    parser.add_argument("--name", "-n", type=str, default=None)
    parser.add_argument("--device", "-d", type=str, default="cpu")
    parser.add_argument("--verbose", "-v", type=int, default=2)
    parser.add_argument("--wandb_sweep", "-ws", type=bool, default=False)
    parser.add_argument("--sync_wandb", "-sw", action="store_true")
    # ============================ Cost ============================= #
    parser.add_argument("--cost_info_str", "-cis", type=str, default="cost")
    # ======================== Environment ========================== #
    parser.add_argument("--train_env_id", "-tei", type=str, default="HalfCheetah-v3")
    parser.add_argument("--eval_env_id", "-eei", type=str, default="HalfCheetah-v3")
    parser.add_argument("--dont_normalize_obs", "-dno", action="store_true")
    parser.add_argument("--dont_normalize_reward", "-dnr", action="store_true")
    parser.add_argument("--seed", "-s", type=int, default=None)
    # ======================== Networks ============================== #
    parser.add_argument("--policy_name", "-pn", type=str, default="MlpPolicy")
    parser.add_argument("--shared_layers", "-sl", type=int, default=None, nargs='*')
    parser.add_argument("--policy_layers", "-pl", type=int, default=[64,64], nargs='*')
    parser.add_argument("--reward_vf_layers", "-rl", type=int, default=[64,64], nargs='*')
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
    # ========================= Losses ============================== #
    parser.add_argument("--clip_range", "-cr", type=float, default=0.2)
    parser.add_argument("--clip_range_reward_vf", "-crv", type=float, default=None)
    parser.add_argument("--ent_coef", "-ec", type=float, default=0.)
    parser.add_argument("--reward_vf_coef", "-rvc", type=float, default=0.5)
    parser.add_argument("--target_kl", "-tk", type=float, default=None)
    parser.add_argument("--max_grad_norm", "-mgn", type=float, default=0.5)
    parser.add_argument("--learning_rate", "-lr", type=float, default=3e-4)
    # =========================== SDE =============================== #
    parser.add_argument("--use_sde", "-us", action="store_true")
    parser.add_argument("--sde_sample_freq", "-ssf", type=int, default=-1)
    # ========================== GAIL ============================ #
    parser.add_argument("--freeze_gail_weights", "-fgw", action="store_true")
    parser.add_argument("--gail_path", "-gp", type=str, default=None)
    parser.add_argument("--learn_cost", "-lc", action="store_true")
    parser.add_argument("--disc_layers", "-dl", type=int, default=[64,64], nargs='*')
    parser.add_argument("--disc_learning_rate", "-dlr", type=float, default=3e-4)
    parser.add_argument("--disc_batch_size", "-dbs", type=int, default=None)
    parser.add_argument('--disc_obs_select_dim', '-dosd', type=int, default=None, nargs='+')
    parser.add_argument('--disc_acs_select_dim', '-dasd', type=int, default=None, nargs='+')
    parser.add_argument('--disc_plot_every', '-dpe', type=int, default=1)
    parser.add_argument('--disc_normalize', '-cn', action='store_true')
    parser.add_argument("--disc_eps", "-de", type=float, default=1e-5)
    parser.add_argument("--clip_obs", "-co", type=int, default=20)
    # ======================== Expert Data ========================== #
    parser.add_argument('--expert_path', '-ep', type=str, default='icrl/expert_data/HCWithPos-vm0')
    parser.add_argument('--expert_rollouts', '-er', type=int, default=20)
    # ======================= Spurious Features ==================== #
    parser.add_argument("--num_spurious_features", "-nsf", type=int, default=None)
    # ===================== Cost Shaping Callback ================== #
    parser.add_argument("--use_cost_shaping_callback", "-ucsc", action="store_true",
                        help="This callback disables GAIL and uses true cost function for shaping.")
    parser.add_argument("--use_cost_net", "-ucn", action="store_true",
                        help="Use a neural network to approximate true cost and use that for shaping.")
    # =============================================================== #
    # =============================================================== #

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
    wandb.init(project=config["project"], name=config["name"], config=config, dir="./gail",
               group=config['group'])
    wandb.config.save_dir = wandb.run.dir
    config = wandb.config

    print(utils.colorize("Configured folder %s for saving" % config.save_dir,
          color="green", bold=True))
    print(utils.colorize("Name: %s" % config.name, color="green", bold=True))

    # Save config
    utils.save_dict_as_json(config.as_dict(), config.save_dir, "config")

    # Train
    gail(config)

    end = time.time()
    print(utils.colorize("Time taken: %05.2f minutes" % ((end-start)/60),
          color="green", bold=True))
