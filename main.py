import argparse
import torch
import time
import os
import pickle
import numpy as np
from gym.spaces import Box, Discrete

from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.MAFTRL import AttentionSACM
import warnings

MSELoss = torch.nn.MSELoss()
warnings.filterwarnings("ignore")


def make_parallel_env(env_id, n_rollout_threads, seed):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=True)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env

        return init_env

    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])


def make_model(env, config):
    if config.model_name == 'maftrl':
        model = AttentionSACM
    else:
        raise NotImplementedError

    model_dir = Path('./models') / config.env_id / config.model_name
    if config.restore:
        if config.load_num is None:
            print("Please input load_num!")
            exit(0)
        else:
            load_dir = model_dir / ('run%i' % int(config.load_num))
    if config.restore and not load_dir.exists():
        print("Load_dir: {} is not exists!".format(load_dir))
        exit(0)
    return model.init_from_env(env,
                               tau=config.tau,
                               pi_lr=config.pi_lr,
                               q_lr=config.q_lr,
                               gamma=config.gamma,
                               pol_hidden_dim=config.pol_hidden_dim,
                               critic_hidden_dim=config.critic_hidden_dim,
                               attend_heads=config.attend_heads,
                               reward_scale=config.reward_scale,
                               max_tolerance=config.max_tolerance) if config.restore is False else model.init_from_save(load_dir / "model.pt", load_critic=True)


def get_reliable_obs(model, obs, recent_obs, tolerance, validity, true_cnt, false_cnt):
    loss = []
    torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])), requires_grad=False)
                 for i in range(model.nagents)]

    out = model.error_detector(torch_obs)

    for i, (osi, oti) in enumerate(zip(torch_obs, out)):
        loss.append(MSELoss(osi, oti).detach())

    for e in range(config.n_rollout_threads):
        for a_i in range(model.nagents):
            loss_i = MSELoss(torch_obs[a_i][e], out[a_i][e]).detach()
            result = False if loss_i > tolerance[a_i] else True
            true_cnt[a_i] += 1 if result == validity[a_i] else 0
            false_cnt[a_i] += 0 if result == validity[a_i] else 1

            if result is False and recent_obs[e][a_i] is not None:
                obs[e][a_i] = recent_obs[e][a_i]
            else:
                recent_obs[e][a_i] = obs[e][a_i]

    loss = []
    torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])), requires_grad=False)
                 for i in range(model.nagents)]

    out = model.error_detector(torch_obs)

    for i, (osi, oti) in enumerate(zip(torch_obs, out)):
        loss.append(MSELoss(osi, oti).detach())

    return obs, loss


def train_detector(env, model, config, logger=None):
    print("Start train ErrorDetector...")

    replay_buffer = ReplayBuffer(config.buffer_length, model.nagents,
                                 [obsp.shape[0] for obsp in env.observation_space],
                                 [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                  for acsp in env.action_space])

    max_tolerance = model.error_detector.realtime_tolerance
    print("max_tolerance: ", max_tolerance)

    t = 0
    t_start = time.time()
    total_loss = [[], [], []]
    predict_true, predict_false = [0, 0, 0], [0, 0, 0]
    model.prep_rollouts(device='cpu')
    for ep_i in range(0, config.n_detector_episodes, config.n_rollout_threads):
        obs, validity = env.reset()
        for et_i in range(config.episode_length):
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])), requires_grad=False)
                         for i in range(model.nagents)]

            torch_agent_actions = model.step(torch_obs, explore=True)
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            next_obs, rewards, dones, infos, next_validity = env.step(actions)

            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            out = model.error_detector(torch_obs)
            loss = []
            for i, (osi, oti) in enumerate(zip(torch_obs, out)):
                error = MSELoss(osi, oti).detach()
                loss.append(error)
                result = True if error < max_tolerance[i] else False
                predict_true[i] += 1 if result == validity[i] else 0
                predict_false[i] += 0 if result == validity[i] else 1
                total_loss[i].append(error)
                if ep_i > 5000:
                    print("label:{}, loss {}: {}, obs: {}".format(validity[i], i, error, torch_obs[i]))
            obs = next_obs
            validity = next_validity
            t += config.n_rollout_threads
            if (len(replay_buffer) >= config.batch_size and
                    (t % config.steps_per_update) < config.n_rollout_threads):

                if config.use_gpu:
                    model.prep_training(device='gpu')
                else:
                    model.prep_training(device='cpu')
                for u_i in range(config.num_updates):
                    sample = replay_buffer.sample(config.batch_size,
                                                  to_gpu=config.use_gpu)
                    model.update_detector(sample, logger=logger)
                model.prep_rollouts(device='cpu')
        if ep_i > 1 and (ep_i + 1) % config.save_interval < config.n_rollout_threads:
            print("Episodes %i of %i: " % (ep_i + config.n_rollout_threads,
                                           config.n_detector_episodes), end='\n')
            for i in range(model.nagents):
                print('Agent %i: mean_total_loss: %f, std_total_loss: %f, predict_correct_rate: %f, time: %f\n' %
                      (i, np.mean(total_loss[i]), np.std(total_loss[i], ddof=1),
                       1. * predict_true[i] / (predict_false[i] + predict_true[i]), round(time.time() - t_start, 6)))

            t_start = time.time()
            total_loss = [[], [], []]
            predict_true, predict_false = [0, 0, 0], [0, 0, 0]

    print("ErrorDetector train completion!\n")


def run(config):
    model_dir = Path('./models') / config.env_id / config.model_name
    if not model_dir.exists():
        run_num = 1
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            run_num = 1
        else:
            run_num = max(exst_run_nums) + 1

    curr_run = 'run%i' % run_num
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))
    env = make_parallel_env(config.env_id, config.n_rollout_threads, run_num)
    model = make_model(env, config)
    train_detector(env, model, config)

    torch.manual_seed(run_num)
    np.random.seed(run_num)
    env = make_parallel_env(config.env_id, config.n_rollout_threads, run_num)
    replay_buffer = ReplayBuffer(config.buffer_length, model.nagents,
                                 [obsp.shape[0] for obsp in env.observation_space],
                                 [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                  for acsp in env.action_space])
    recent_reliable_obs = [[None for i in range(model.nagents)] for e in range(config.n_rollout_threads)]

    total_loss = [[], [], []]
    predict_true_cnt = [0, 0, 0]
    predict_false_cnt = [0, 0, 0]

    print("Start train Agents...")

    t = 0
    steps, avg_ep_rew = 0, 0
    t_start = time.time()

    each_rws = []
    large_rws = []
    small_rws = []

    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):

        model.prep_rollouts(device='cpu')
        obs, validity = env.reset()
        obs, loss = get_reliable_obs(model, obs, recent_reliable_obs, model.error_detector.realtime_tolerance, validity,
                                     predict_true_cnt, predict_false_cnt)
        for i, l in enumerate(loss):
            total_loss[i].append(l)

        for et_i in range(config.episode_length):
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])), requires_grad=False)
                         for i in range(model.nagents)]

            # get actions as torch Variables
            torch_agent_actions = model.step(torch_obs, explore=True)

            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]

            next_obs, rewards, dones, infos, next_validity = env.step(actions)

            next_obs, loss = get_reliable_obs(model, next_obs, recent_reliable_obs,
                                              model.error_detector.realtime_tolerance, next_validity, predict_true_cnt,
                                              predict_false_cnt)
            for i, l in enumerate(loss):
                total_loss[i].append(l)

            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            obs = next_obs
            validity = next_validity
            t += config.n_rollout_threads
            if (len(replay_buffer) >= config.batch_size and
                    (t % config.steps_per_update) < config.n_rollout_threads):

                if config.use_gpu:
                    model.prep_training(device='gpu')
                else:
                    model.prep_training(device='cpu')
                for u_i in range(config.num_updates):
                    sample = replay_buffer.sample(config.batch_size,
                                                  to_gpu=config.use_gpu)
                    model.update_detector(sample, logger=logger)
                    model.update_critic(sample, logger=logger)
                    model.update_policies(sample, logger=logger)
                    model.update_all_targets()
                model.prep_rollouts(device='cpu')
        ep_rews = replay_buffer.get_average_rewards(
            config.episode_length * config.n_rollout_threads)

        steps += 1

        for a_i, a_ep_rew in enumerate(ep_rews):
            avg_ep_rew += a_ep_rew
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)
        each_rws.append(ep_rews[0])
        small_rws.append(sum(ep_rews))
        large_rws.append(sum(ep_rews) * config.episode_length)
        logger.add_scalar('large_rewards', large_rws[-1], ep_i)
        logger.add_scalar('small_rewards', small_rws[-1], ep_i)

        if ep_i > 1 and (ep_i + 1) % config.save_interval < config.n_rollout_threads:
            print("Episodes %i of %i" % (ep_i + config.n_rollout_threads,
                                         config.n_episodes), end=' ')
            print('mean_episode_rewards: %f, time: %f' % (
                avg_ep_rew / steps * config.episode_length, round(time.time() - t_start, 3)))
            for i in range(model.nagents):
                print('Agent %i:  mean_total_loss: %f, std_total_loss: %f, predict_correct_rate: %f, time: %f\n' %
                      (i, np.mean(total_loss[i]), np.std(total_loss[i], ddof=1),
                       1. * predict_true_cnt[i] / (predict_false_cnt[i] + predict_true_cnt[i]), round(time.time() - t_start, 6)))

            t_start = time.time()
            steps, avg_ep_rew = 0, 0
            total_loss = [[], [], []]
            predict_true_cnt = [0, 0, 0]
            predict_false_cnt = [0, 0, 0]
            model.prep_rollouts(device='cpu')
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            if (ep_i + 1) % (config.save_interval * 5) < config.n_rollout_threads:
                model.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
                model.save(run_dir / 'model.pt')

    large_rew_file_name = log_dir / (config.model_name + '_large_rewards.pkl')
    with open(large_rew_file_name, 'wb') as fp:
        pickle.dump(large_rws, fp)

    small_rew_file_name = log_dir / (config.model_name + '_small_rewards.pkl')
    with open(small_rew_file_name, 'wb') as fp:
        pickle.dump(small_rws, fp)

    each_rew_file_name = log_dir / (config.model_name + '_each_rewards.pkl')
    with open(each_rew_file_name, 'wb') as fp:
        pickle.dump(each_rws, fp)

    model.save(run_dir / 'model.pt')
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()

    print("Agents train completion!\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_name", help="Name of algorithm")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=20000, type=int)
    parser.add_argument("--n_detector_episodes", default=5000, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--num_updates", default=4, type=int)
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--pol_hidden_dim", default=128, type=int)
    parser.add_argument("--critic_hidden_dim", default=128, type=int)
    parser.add_argument("--attend_heads", default=4, type=int)
    parser.add_argument("--pi_lr", default=0.001, type=float)
    parser.add_argument("--q_lr", default=0.001, type=float)
    parser.add_argument("--tau", default=0.001, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--reward_scale", default=100., type=float)
    parser.add_argument("--max_tolerance", default=0.2, type=float)
    parser.add_argument("--use_gpu", action='store_true', default=False)
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--load_num", default=None, type=int)
    parser.add_argument("--display", action="store_true", default=False)

    config = parser.parse_args()

    run(config)
