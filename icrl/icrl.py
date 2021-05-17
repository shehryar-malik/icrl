import argparse
import importlib
import json
import os
import pickle
import sys
import time

import gym
import numpy as np
from stable_baselines3 import PPOLagrangian
from stable_baselines3.common import logger
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import sync_envs_normalization, VecNormalize
from tqdm import tqdm

import icrl.utils as utils
from icrl.plot_utils import plot_obs_ant
from icrl.exploration import ExplorationRewardCallback, LambdaShapingCallback
import wandb
from icrl.constraint_net import ConstraintNet, plot_constraints
from icrl.true_constraint_net import get_true_cost_function, null_cost


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

def icrl(config):
    # We only want to use cost wrapper for custom environments
    use_cost_wrapper_train = True
    use_cost_wrapper_eval = False

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

    # We don't need cost when taking samples
    sampling_env = utils.make_eval_env(env_id=config.train_env_id,
                                       use_cost_wrapper=False,
                                       normalize_obs=not config.dont_normalize_obs)

    eval_env = utils.make_eval_env(env_id=config.eval_env_id,
                                   use_cost_wrapper=use_cost_wrapper_eval,
                                   normalize_obs=not config.dont_normalize_obs)

    # Set specs
    is_discrete = isinstance(train_env.action_space, gym.spaces.Discrete)
    obs_dim = train_env.observation_space.shape[0]
    acs_dim = train_env.action_space.n if is_discrete else train_env.action_space.shape[0]
    action_low, action_high = None, None
    if isinstance(sampling_env.action_space, gym.spaces.Box):
        action_low, action_high = sampling_env.action_space.low, sampling_env.action_space.high

    # Load expert data
    (expert_obs, expert_acs), expert_mean_reward = load_expert_data(config.expert_path, config.expert_rollouts)
    expert_agent = PPOLagrangian.load(os.path.join(config.expert_path, "files/best_model.zip"))

    # Logger
    icrl_logger = logger.HumanOutputFormat(sys.stdout)

    # Initialize constraint net, true constraint net
    cn_lr_schedule = lambda x : (config.anneal_clr_by_factor**(config.n_iters*(1 - x))) * config.cn_learning_rate
    constraint_net = ConstraintNet(
            obs_dim,
            acs_dim,
            config.cn_layers,
            config.cn_batch_size,
            cn_lr_schedule,
            expert_obs,
            expert_acs,
            is_discrete,
            config.cn_reg_coeff,
            config.cn_obs_select_dim,
            config.cn_acs_select_dim,
            no_importance_sampling=config.no_importance_sampling,
            per_step_importance_sampling=config.per_step_importance_sampling,
            clip_obs=config.clip_obs,
            initial_obs_mean=None if not config.cn_normalize else np.zeros(obs_dim),
            initial_obs_var=None if not config.cn_normalize else np.ones(obs_dim),
            action_low=action_low,
            action_high=action_high,
            target_kl_old_new=config.cn_target_kl_old_new,
            target_kl_new_old=config.cn_target_kl_new_old,
            train_gail_lambda=config.train_gail_lambda,
            eps=config.cn_eps,
            device=config.device
    )

    # Pass constraint net cost function to cost wrapper (train env)
    train_env.set_cost_function(constraint_net.cost_function)

    true_cost_function = get_true_cost_function(config.eval_env_id)

    # Setup plotting for constraint net
    if config.clip_obs is not None:
        obs = np.clip(expert_obs, -config.clip_obs, config.clip_obs)
    else:
        obs = expert_obs

    cn_plot_dir = os.path.join(config.save_dir, "constraint_net")
    utils.del_and_make(cn_plot_dir)

    other_plot_dir = os.path.join(config.save_dir, "plots")
    utils.del_and_make(other_plot_dir)

    # Plot True cost function and expert samples
    plot_constraints(
            true_cost_function, eval_env, config.eval_env_id, constraint_net.select_dim,
            obs_dim, acs_dim, os.path.join(config.save_dir, "true_constraint_net.png"),
            observations=obs
    )

    # Initialize agent
    create_nominal_agent = lambda: PPOLagrangian(
            policy=config.policy_name,
            env=train_env,
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
            budget=config.budget,
            seed=config.seed,
            device=config.device,
            verbose=0,
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
    nominal_agent = create_nominal_agent()

    # Callbacks
    all_callbacks = []
    if config.use_curiosity_driven_exploration:
        explorationCallback = ExplorationRewardCallback(obs_dim, acs_dim, device=config.device)
        all_callbacks.append(explorationCallback)

    # Warmup
    timesteps = 0.
    if config.warmup_timesteps is not None:
        print(utils.colorize("\nWarming up", color="green", bold=True))
        with utils.ProgressBarManager(config.warmup_timesteps) as callback:
            nominal_agent.learn(total_timesteps=config.warmup_timesteps,
                                cost_function=null_cost, # During warmup we dont want to incur any cost
                                callback=callback)
            timesteps += nominal_agent.num_timesteps

    # Train
    start_time = time.time()
    print(utils.colorize("\nBeginning training", color="green", bold=True), flush=True)
    best_true_reward, best_true_cost, best_forward_kl, best_reverse_kl = -np.inf, np.inf, np.inf, np.inf
    for itr in range(config.n_iters):
        if config.reset_policy and itr != 0:
            print(utils.colorize("Resetting agent", color="green", bold=True), flush=True)
            nominal_agent = create_nominal_agent()
        current_progress_remaining = 1-float(itr)/float(config.n_iters)

        # Update agent
        with utils.ProgressBarManager(config.forward_timesteps) as callback:
            nominal_agent.learn(
                    total_timesteps=config.forward_timesteps,
                    cost_function="cost",         # Cost should come from cost wrapper
                    callback=[callback]+all_callbacks
            )
            forward_metrics = logger.Logger.CURRENT.name_to_value
            timesteps += nominal_agent.num_timesteps

        # Sample nominal trajectories
        sync_envs_normalization(train_env, sampling_env)
        orig_observations, observations, actions, rewards, lengths = utils.sample_from_agent(
                nominal_agent, sampling_env, config.expert_rollouts)

        # Plot constraint net periodically
        if itr % config.cn_plot_every == 0:
            obs_for_plot = orig_observations
            if config.clip_obs is not None:
                obs_for_plot = np.clip(obs_for_plot, -config.clip_obs, config.clip_obs)
            plot_constraints(constraint_net.cost_function, eval_env, config.eval_env_id, constraint_net.select_dim,
                    obs_dim, acs_dim, os.path.join(cn_plot_dir, "%d.png"%itr), observations=obs_for_plot)

            if config.train_env_id =='AntWall-v0':
                plot_obs_ant(obs_for_plot, os.path.join(other_plot_dir, "%d.png"%itr))

        # Update constraint net
        mean, var = None, None
        if config.cn_normalize:
            mean, var = sampling_env.obs_rms.mean, sampling_env.obs_rms.var
        backward_metrics = constraint_net.train(config.backward_iters, orig_observations, actions, lengths,
                                                mean, var, current_progress_remaining)

        # Pass updated cost_function to cost wrapper (train_env)
        train_env.set_cost_function(constraint_net.cost_function)

        # Evaluate:
        # (1): True cost on nominal environment
        average_true_cost = np.mean(true_cost_function(orig_observations, actions))
        samples_behind = np.mean(orig_observations[...,0] < -3)
        samples_infront = np.mean(orig_observations[...,0] > 3)
        # (2): Reward on true environment
        sync_envs_normalization(train_env, eval_env)
        average_true_reward, std_true_reward = evaluate_policy(nominal_agent, eval_env, n_eval_episodes=10,
                                                               deterministic=False)
        # (3): KLs
        forward_kl = utils.compute_kl(nominal_agent, expert_obs, expert_acs, expert_agent)
        reverse_kl = utils.compute_kl(expert_agent, orig_observations, actions, nominal_agent)

        # Save:
        # (1): Periodically
        if itr % config.save_every == 0:
            path = os.path.join(config.save_dir, f"models/icrl_{itr}_itrs")
            utils.del_and_make(path)
            nominal_agent.save(os.path.join(path, f"nominal_agent"))
            constraint_net.save(os.path.join(path, f"cn.pt"))
            if isinstance(train_env, VecNormalize):
                train_env.save(os.path.join(path, f"{itr}_train_env_stats.pkl"))
        # (2): Best model
        if average_true_reward > best_true_reward:
            print(utils.colorize("Saving new best model", color="green", bold=True), flush=True)
            nominal_agent.save(os.path.join(config.save_dir, "best_nominal_model"))
            constraint_net.save(os.path.join(config.save_dir, "best_cn_model.pt"))
            if isinstance(train_env, VecNormalize):
                train_env.save(os.path.join(config.save_dir, "train_env_stats.pkl"))

        # Update best metrics
        if average_true_reward > best_true_reward:
            best_true_reward = average_true_reward
        if average_true_cost < best_true_cost:
            best_true_cost = average_true_cost
        if forward_kl < best_forward_kl:
            best_forward_kl = forward_kl
        if reverse_kl < best_reverse_kl:
            best_reverse_kl = reverse_kl

        # Collect metrics
        metrics = {
                "time(m)": (time.time()-start_time)/60,
                "iteration": itr,
                "timesteps": timesteps,
                "true/reward": average_true_reward,
                "true/reward_std": std_true_reward,
                "true/cost": average_true_cost,
                "true/samples_infront": samples_infront,
                "true/samples_behind": samples_behind,
                "true/forward_kl": forward_kl,
                "true/reverse_kl": reverse_kl,
                "best_true/best_reward": best_true_reward,
                "best_true/best_cost": best_true_cost,
                "best_true/best_forward_kl": best_forward_kl,
                "best_true/best_reverse_kl": best_reverse_kl
                }
        metrics.update({k.replace("train/", "forward/"): v for k, v in forward_metrics.items()})
        metrics.update(backward_metrics)

        # Log
        if config.verbose > 0:
            icrl_logger.write(metrics, {k: None for k in metrics.keys()}, step=itr)
        wandb.log(metrics)

    # Make video of final model
    if not config.wandb_sweep:
        sync_envs_normalization(train_env, eval_env)
        utils.eval_and_make_video(eval_env, nominal_agent, config.save_dir, "final_policy")

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
    parser.add_argument("--device", "-d", type=str, default="cpu")
    parser.add_argument("--verbose", "-v", type=int, default=2)
    parser.add_argument("--sync_wandb", "-sw", action="store_true")
    parser.add_argument("--wandb_sweep", "-ws", type=bool, default=False)
    # ======================== Environments ========================= #
    parser.add_argument("--train_env_id", "-tei", type=str, default="HalfCheetah-v3")
    parser.add_argument("--eval_env_id", "-eei", type=str, default="HalfCheetah-v3")
    parser.add_argument("--dont_normalize_obs", "-dno", action="store_true")
    parser.add_argument("--dont_normalize_reward", "-dnr", action="store_true")
    parser.add_argument("--dont_normalize_cost", "-dnc", action="store_true")
    parser.add_argument("--seed", "-s", type=int, default=None)
    parser.add_argument("--clip_obs", "-co", type=int, default=20)
    # ============================ Cost ============================= #
    parser.add_argument("--cost_info_str", "-cis", type=str, default="cost")

    # ========================= Policy =============================== #
    # ======================== Networks ============================== #
    parser.add_argument("--policy_name", "-pn", type=str, default="TwoCriticsMlpPolicy")
    parser.add_argument("--shared_layers", "-sl", type=int, default=None, nargs='*')
    parser.add_argument("--policy_layers", "-pl", type=int, default=[64,64], nargs='*')
    parser.add_argument("--reward_vf_layers", "-rvl", type=int, default=[64,64], nargs='*')
    parser.add_argument("--cost_vf_layers", "-cvl", type=int, default=[64,64], nargs='*')
    # ========================= Training ============================ #
    parser.add_argument("--n_steps", "-ns", type=int, default=2048)
    parser.add_argument("--batch_size", "-bs", type=int, default=64)
    parser.add_argument("--n_epochs", "-ne", type=int, default=10)
    parser.add_argument("--num_threads", "-nt", type=int, default=5)
    parser.add_argument("--save_every", "-se", type=float, default=1)
    parser.add_argument("--eval_every", "-ee", type=float, default=2048)
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
                        help="Sets Learning Rate of Dual Variables if use_pid is not true.")
    # ======================= Exploration =========================== #
    parser.add_argument("--use_sde", "-us", action="store_true")
    parser.add_argument("--use_curiosity_driven_exploration", "-ucde", action="store_true")
    parser.add_argument("--sde_sample_freq", "-ssf", type=int, default=-1)
    # =============================================================== #
    # =============================================================== #

    # ========================== ICRL =============================== #
    parser.add_argument('--train_gail_lambda', '-tgl', action='store_true')
    parser.add_argument("--n_iters", "-ni", type=int, default=100)
    parser.add_argument("--warmup_timesteps", "-wt", type=lambda x: int(float(x)), default=None)
    parser.add_argument("--forward_timesteps", "-ft", type=lambda x: int(float(x)), default=1e6)
    parser.add_argument("--backward_iters", "-bi", type=int, default=10)
    parser.add_argument('--no_importance_sampling', '-nis', action='store_true')
    parser.add_argument('--per_step_importance_sampling', '-psis', action='store_true')
    parser.add_argument('--reset_policy', '-rp', action='store_true')
    # ====================== Constraint Net ========================= #
    parser.add_argument("--cn_layers", "-cl", type=int, default=[64,64], nargs='*')
    parser.add_argument("--anneal_clr_by_factor", "-aclr", type=float, default=1.0)
    parser.add_argument("--cn_learning_rate", "-clr", type=float, default=3e-4)
    parser.add_argument("--cn_reg_coeff", "-crc", type=float, default=0)
    parser.add_argument("--cn_batch_size", "-cbs", type=int, default=None)
    parser.add_argument('--cn_obs_select_dim', '-cosd', type=int, default=None, nargs='+')
    parser.add_argument('--cn_acs_select_dim', '-casd', type=int, default=None, nargs='+')
    parser.add_argument('--cn_plot_every', '-cpe', type=int, default=1)
    parser.add_argument('--cn_normalize', '-cn', action='store_true')
    parser.add_argument("--cn_target_kl_old_new", "-ctkon", type=float, default=10)
    parser.add_argument("--cn_target_kl_new_old", "-ctkno", type=float, default=10)
    parser.add_argument("--cn_eps", "-ce", type=float, default=1e-5)
    # ======================== Expert Data ========================== #
    parser.add_argument('--expert_path', '-ep', type=str, default='icrl/expert_data/HCWithPos-vm0')
    parser.add_argument('--expert_rollouts', '-er', type=int, default=20)
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
    wandb.init(project=config["project"], name=config["name"], config=config, dir="./icrl",
               group=config["group"])
    wandb.config.save_dir = wandb.run.dir
    config = wandb.config

    print(utils.colorize("Configured folder %s for saving" % config.save_dir,
        color="green", bold=True))
    print(utils.colorize("Name: %s" % config.name, color="green", bold=True))

    # Save config
    utils.save_dict_as_json(config.as_dict(), config.save_dir, "config")

    # Train
    icrl(config)

    end = time.time()
    print(utils.colorize("Time taken: %05.2f hours" % ((end-start)/3600),
        color="green", bold=True))

if __name__=='__main__':
    main()
