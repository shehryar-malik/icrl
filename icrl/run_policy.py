"""Load and run policy"""

import argparse
import os
import shutil

import numpy as np
from stable_baselines3 import PPOLagrangian
from stable_baselines3.common.vec_env import VecNormalize

import icrl.utils as utils
import wandb

USER="spiderbot"

def load_config(d):
    config = utils.load_dict_from_json(d, "config")
    config = utils.dict_to_namespace(config)
    return config

def run_policy(args):
    # Find which file to load
    if args.is_icrl:
        if args.load_itr is not None:
            f = "model/icrl_{args.load_itr}_its/nominal_agent"
        else:
            f = "best_nominal_model"
    else:
        if args.load_itr is not None:
            f = f"models/rl_model_{args.load_itr}_steps"
        else:
            f = "best_model"

    # Configure paths (restore from W&B server if needed)
    if args.remote:
        # Save everything in wandb/remote/<run_id>
        load_dir = os.path.join("icrl/wandb/remote/", args.load_dir.split('/')[-1])
        utils.del_and_make(load_dir)
        # Restore form W&B
        wandb.init(dir=load_dir)
        run_path = os.path.join(USER, args.load_dir)
        wandb.restore("config.json", run_path=run_path, root=load_dir)
        config = load_config(load_dir)
        if not config.dont_normalize_obs:
            wandb.restore("train_env_stats.pkl", run_path=run_path, root=load_dir)
        wandb.restore(f+".zip", run_path=run_path, root=load_dir)
    else:
        load_dir = os.path.join(args.load_dir, "files")
        config = load_config(load_dir)

    save_dir = os.path.join(load_dir, args.save_dir)
    utils.del_and_make(save_dir)
    model_path = os.path.join(load_dir, f)

    # Load model
    model = PPOLagrangian.load(model_path)

    # Create env, model
    def make_env():
        env_id = args.env_id or config.eval_env_id
        env = utils.make_eval_env(env_id, use_cost_wrapper=False, normalize_obs=False)

        # Restore enviroment stats
        if not config.dont_normalize_obs:
            env = VecNormalize.load(os.path.join(load_dir, "train_env_stats.pkl"), env)
            env.norm_reward = False
            env.training = False

        return env

    # Evaluate and make video
    if not args.dont_make_video:
        env = make_env()
        utils.eval_and_make_video(env, model, save_dir, "video", args.n_rollouts)

    # Check if we want to save using airl scheme
    if args.save_using_airl_scheme:
        sampling_func = utils.sample_from_agent_airl
    else:
        sampling_func = utils.sample_from_agent

    if not args.dont_save_trajs:
        env = make_env()
        # Make saving dir
        rollouts_dir = os.path.join(save_dir, "rollouts")
        utils.del_and_make(rollouts_dir)
        idx = 0
        while True:
            saving_dict = sampling_func(model, env, 1)
            if not args.save_using_airl_scheme:
                observations, _, actions, rewards, lengths = saving_dict
                saving_dict = dict(observations=observations, actions=actions, rewards=rewards, lengths=lengths)
                saving_dict['save_scheme'] = 'not_airl'
            else:
                saving_dict['save_scheme'] = 'airl'
            if (args.reward_threshold is None or np.mean(saving_dict['rewards']) >= args.reward_threshold) and\
               (args.length_threshold is None or np.mean(saving_dict['lengths']) >= args.length_threshold):
                print(f"{idx}. Mean reward: {np.mean(saving_dict['rewards'])} | Mean length: {np.mean(saving_dict['lengths'])}")
                utils.save_dict_as_pkl(saving_dict,
                                       rollouts_dir, str(idx))
                idx += 1
                if idx == args.n_rollouts:
                    break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_to_run", type=str)
    parser.add_argument("--load_dir", "-l", type=str, default="icrl/wandb/latest-run/")
    parser.add_argument("--is_icrl", "-ii", action='store_true')
    parser.add_argument("--remote", "-r", action="store_true")
    parser.add_argument("--save_dir", "-s", type=str, default="run_policy")
    parser.add_argument("--env_id", "-e", type=str, default=None)
    parser.add_argument("--load_itr", "-li", type=int, default=None)
    parser.add_argument("--n_rollouts", "-nr", type=int, default=3)
    parser.add_argument("--dont_make_video", "-dmv", action="store_true")
    parser.add_argument("--dont_save_trajs", "-dst", action="store_true")
    parser.add_argument("--save_using_airl_scheme", "-suas", action="store_true")
    parser.add_argument("--reward_threshold", "-rt", type=float, default=None)
    parser.add_argument("--length_threshold", "-lt", type=int, default=None)
    args = parser.parse_args()

    run_policy(args)
