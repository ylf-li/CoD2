import numpy as np
import torch
import gym
import argparse
import os
import time

from utils import utils
from lib import  TD3,TD2,TADD,CoD2, DoubleTD3,TD33,QMO
import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings("ignore")


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)
    states_data = []
    actions_data = []
    rewards = []
    avg_reward = 0.
    for _ in range(1):
        state, done = eval_env.reset(), False
        while not done:
            states_data.append(state)

            action = policy.select_action(np.array(state))
            actions_data.append(action)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
            rewards.append(reward)

    avg_reward /= eval_episodes

    # print("---------------------------------------")
    # print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    # print("---------------------------------------")
    return avg_reward

def run_mujoco(policy,env,seed,w1,w2):

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=policy)  # Policy name (TD30.002, DDPG or OurDDPG)
    parser.add_argument("--env", default=env)  # OpenAI gym environment name
    parser.add_argument("--seed", default=seed, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e4, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=1e4, type=int)  # How often (time steps) we evaluate
    # parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment
    parser.add_argument("--max_timesteps", default=2e4, type=int)  # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)  # Std of Gaussian explopiration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--w1", default=w1)  # Target network update rate
    parser.add_argument("--w2", default=w2)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true",default=True)  # Save model and optimizer parameters
    parser.add_argument("--load_model",default="")  # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.w1}")
    print("---------------------------------------")

    if not os.path.exists("results"):
        os.makedirs("results")

    policy_name = policy+'-w1='+str(w1)+'-w2='+str(w2)
    # policy_name = policy

    model_path = f"/media/ly/data8/log/COTD3logs/modelesinit/{policy_name}"
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # if args.save_model and not os.path.exists(policy):
    #     os.makedirs(policy)

    logpath = os.path.join('/opt/code/TD3/run3',policy_name,
                           policy+'-'+args.env+'-seed='+str(args.seed))
    # print(logpath)
    writer = SummaryWriter(logpath)

    data_state = os.path.join('/media/ly/data8/log/COTD3logs/samples',logpath)

    if not os.path.exists(data_state):
        os.makedirs(data_state)

    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        # 'w1':args.w1,
        # 'w2':args.w2
    }

    # Initialize policy

    kwargs["policy_noise"] = args.policy_noise * max_action
    kwargs["noise_clip"] = args.noise_clip * max_action
    kwargs["policy_freq"] = args.policy_freq
    # kwargs["w"] = args.w

    if policy=='DoubleDDPGTD3':
        print('---------------------------runing TD3-----------------------')
        policy = DoubleTD3.DoubleTD3(**kwargs)
    if policy=='TD3':
        print('---------------------------runing TD3-----------------------')
        policy = TD33.TD3(**kwargs)
    elif policy == 'TD2':
        print('--------------------------runing TD2----------------------')
        policy = TD2.TD2(**kwargs)
    elif policy == 'TADD':
        print('--------------------------runing TADD----------------------')
        policy = TADD.TADD(**kwargs)
    elif policy == 'CoD2':
        print('--------------------------runing CoD2----------------------')
        policy = CoD2.CoD2(**kwargs)
    elif policy == 'QMO':
        print('--------------------------runing QMO----------------------')
        policy = QMO.QMO(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    # Evaluate untrained policy
    evaluations = [eval_policy(policy, args.env, args.seed)]

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(int(args.max_timesteps)):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                    policy.select_action(np.array(state))
                    + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            # print(
            #     f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            writer.add_scalar('reward', episode_reward, global_step=t)
            # writer.add_scalar('percent/percent', policy.percent, global_step=t)
            # writer.add_scalar('percent/diff', policy.diff, global_step=t)
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            reward = eval_policy(policy, args.env, args.seed)
            evaluations.append(reward)
            writer.add_scalar('reward_eval', reward, global_step=t)

            np.save(f"results/{file_name}", evaluations)

            # policy.save(f"/media/ly/data8/log/COTD3logs/modelesinit/{policy_name}/{file_name}_{t}")

games = ['Swimmer-v2','Walker2d-v2','HalfCheetah-v2']
games = ['Ant-v2','Hopper-v2','BipedalWalker-v2']
games = ['Swimmer-v2']

def train(games,policy,seeds,w1, w2):

    process = []
    for env in games:
        for seed in seeds:
            p = mp.Process(target=run_mujoco,args=(policy,env,seed,w1, w2))
            p.start()
            process.append(p)
        for p in process:
            p.join()

from  gym.envs.mujoco import SwimmerEnv
from  gym.envs.mujoco import HopperEnv
id = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(id)
# if __name__=='__main__':
#     mp.set_start_method('spawn')
#     train(games, policy='TD3', seeds=[0,1,2,3,4], w1 =1, w2 =1)
#     train(games, policy='CoD2', seeds=[0,1,2,3,4], w1 =1, w2 =1)
#
games = ['HalfCheetah-v2','Ant-v2','Hopper-v2','Swimmer-v2','Reacher-v2','Walker2d-v2',
         'LunarLanderContinuous-v2','BipedalWalker-v2','InvertedDoublePendulum-v2',]
for env in games:
    time1 = time.time()
    run_mujoco(policy='QMO',env=env,seed=0,w1=0.6, w2=1000)
    time2 = time.time()
    print(env, time2-time1)