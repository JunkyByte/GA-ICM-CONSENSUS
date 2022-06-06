import ray
from ray.util import ActorPool
import pretty_errors
import gym
import time
from gym.wrappers import ResizeObservation, TimeLimit
import torch
from torch.nn.functional import mse_loss, cross_entropy
import numpy as np
import sys
from gym.spaces import Box
import os
from ga.ga import rank_selection, lw_gaussian_noise, GA, Individual, lw_crossover, consensus_crossover, Param
from icm import ICM, Memory
from model import Net
from atari_wrap import MainGymWrapper
from functools import partial
import pyglet
from scipy.special import softmax
pretty_errors.configure(
    separator_character = '*',
    filename_display    = pretty_errors.FILENAME_EXTENDED,
    line_number_first   = True,
    display_link        = True,
    lines_before        = 5,
    lines_after         = 2,
    line_color          = pretty_errors.RED + '> ' + pretty_errors.default_config.line_color,
    code_color          = '  ' + pretty_errors.default_config.line_color,
    truncate_code       = True,
    display_locals      = True
)
pyglet.options["debug_gl"] = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('**** PYTORCH DEVICE: %s ****' % device)


class Net_Individual(Individual):
    def initialize_genes(self):
        self.net = Net(input_shape=observation_space, output_size=action_space)

        self.genes = []
        for name, param in self.net.named_parameters():
            self.genes.append(param.data.numpy())

        self.genes_to_weights()
        self.net.to(device)

    def predict(self, x):
        if not isinstance(x, torch.Tensor):
            if x.ndim == 3:
                x = torch.tensor(x.copy(), dtype=torch.float32).permute(2, 0, 1)
            else:
                x = torch.tensor(x, dtype=torch.float32)

        if x.ndim == 1 or x.ndim == 3:
            x = x[None]

        pred = self.net(x.to(device))
        if self.net.is_discrete:
            #return np.argmax(pred, axis=1), pred
            return [np.random.choice(pred.shape[1], p=softmax(pred)[0])], pred
        return pred, pred

    def gene_size(self):
        size = 0
        for param in self.net.parameters():
            size += np.prod(param.shape)
        return size

    def genes_to_weights(self):
        for i, param in enumerate(self.net.parameters()):
            param.data = torch.tensor(self.genes[i], dtype=torch.float).to(device)

    def weights_to_genes(self):
        self.genes = []
        for i, param in enumerate(self.net.parameters()):
            self.genes.append(param.data.cpu().numpy())

    def get_weights(self):
        self.w = []
        for i, param in enumerate(self.net.parameters()):
            self.w.append(param.data.cpu().numpy())
        return self.w

    def train(self, memory):
        state, _, action = memory
        idx = np.random.randint(0, high=len(state), size=min(len(state), 64))

        state = np.array(state)[idx]
        action = np.array(action)[idx]

        state = torch.tensor(state, dtype=torch.float).detach().to(device)
        action = torch.tensor(action, dtype=torch.float).detach().to(device)
        a_hat = self.net.forward(state, train=True)

        if self.net.is_discrete:
            a_hat = a_hat / a_hat.norm(dim=1).detach().view(-1, 1)
            action = action.squeeze(1) / action.norm(dim=1).detach().view(-1, 1)
            loss = mse_loss(a_hat, action)
        else:
            loss = mse_loss(input=a_hat, target=action.squeeze(1))

        loss.backward()

        self.net.opt.step()
        return loss.cpu().detach().numpy()


class NegativeRewardEnd(gym.Wrapper):
    def __init__(self, env, start_min=0, end_reward=0):
        super(NegativeRewardEnd, self).__init__(env)
        self._tot_reward = 0
        self._max_reward = 0
        self._end_reward = end_reward
        self._start_min = start_min

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._tot_reward += reward

        if reward > 0:
            self._max_reward = self._tot_reward

        if self._tot_reward < self._max_reward + self._end_reward and self._max_reward > self._start_min:
            done = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._tot_reward = 0
        self._max_reward = 0
        return self.env.reset(**kwargs)


class StackChannels(gym.Wrapper):
    def __init__(self, env, num_frames=4):
        super(StackChannels, self).__init__(env)
        self.num_frames = num_frames
        self.dtype = self.env.observation_space.dtype
        self.shape = self.env.observation_space.shape
        self.buffer = np.zeros((self.num_frames, *self.shape), dtype=self.dtype)
        low = np.repeat(self.observation_space.low, num_frames, axis=2)
        high = np.repeat(self.observation_space.high, num_frames, axis=2)
        self.observation_space = Box(low=low, high=high, dtype=self.dtype)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.buffer = self.concat(observation)[-self.num_frames:]
        return np.concatenate(self.buffer, axis=-1), reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.buffer = np.zeros((self.num_frames, *self.shape), dtype=self.dtype)
        self.buffer = self.concat(observation)[-self.num_frames:]
        return np.concatenate(self.buffer, axis=-1)

    def concat(self, obs):
        return np.concatenate([self.buffer, obs[None]], axis=0)


class SkipWrapper(gym.Wrapper):
    def __init__(self, env, repeat_count):
        super(SkipWrapper, self).__init__(env)
        self.repeat_count = repeat_count
        self.stepcount = 0

    def step(self, action):
        done = False
        total_reward = 0
        current_step = 0
        while current_step < (self.repeat_count + 1) and not done:
            self.stepcount += 1
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            current_step += 1
        return obs, total_reward, done, info

    def reset(self):
        self.stepcount = 0
        return self.env.reset()


def apply_wrappers(env, env_id, FRAME_SKIP, FRAME_STACK):
    if env_id == 'gridworld-v0':
        env = TimeLimit(env, max_episode_steps=128)
        env = ResizeObservation(env, shape=32)
    if env_id == 'CarRacing-v0':
        env = env.env  # Disable timelimit
        env = SkipWrapper(env, repeat_count=FRAME_SKIP - 1)
        env = NegativeRewardEnd(env, end_reward=-25)
        env = StackChannels(env, num_frames=FRAME_STACK)
        env.unwrapped.hardcore = True
        env.unwrapped.hardcore_speed = True
        env.unwrapped.verbose = 0
    if env_id == 'BipedalWalker-v3':
        env = NegativeRewardEnd(env, start_min=0.5, end_reward=-1)
    if 'Breakout' in env_id:
        env = MainGymWrapper.wrap(env)
    if 'Pong' in env_id:
        env = MainGymWrapper.wrap(env)
    if 'Pacman' in env_id:
        env = MainGymWrapper.wrap(env, is_pacman=True)
    return env


if __name__ == '__main__':
    # Environment
    FRAME_SKIP = 2  # 1 for disabled
    FRAME_STACK = 4
    finish_reward = {'CartPole-v1': 500, 'Acrobot-v1': -90, 'MountainCar-v0': -90, 'BreakoutDeterministic-v0': 1000,
                     'LunarLander-v2': 1000, 'CarRacing-v0': 1500, 'PongDeterministic-v0': 1000,
                     'MsPacmanDeterministic-v0': 1000, 'LunarLanderContinuous-v2': 1000, 'BipedalWalker-v3': 1000}
    env_id = 'MsPacmanDeterministic-v0'
    env = gym.make(env_id)
    env = apply_wrappers(env, env_id, FRAME_SKIP, FRAME_STACK)

    action_space = env.action_space
    observation_space = env.observation_space

    # Calculate gene_size
    net = Net(input_shape=observation_space, output_size=action_space)
    size = len(list(net.parameters()))

    # Configs
    pop_size = 100
    gene_size = size
    selection_size = Param(75, name='Sel Size')
    elite_size = Param(10, name='Elite Size')
    expl_elite = 0
    consensus_step = None
    USE_EXPL = True if expl_elite != 0 else False
    if not USE_EXPL:
        top_expl = None

    is_ssh = "SSH_CONNECTION" in os.environ

    # Logging
    LOGGING_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'log/%s/' % env_id)
    if not os.path.isdir(LOGGING_FOLDER):
        os.mkdir(LOGGING_FOLDER)

    START_STEP_LOG = 0

    # Setup Genetic Algorithm class
    ga = GA(pop_size, Net_Individual, gene_size, selection_size, rank_selection, lw_crossover,
            lw_gaussian_noise, LOGGING_FOLDER, log_step=START_STEP_LOG, elite_size=elite_size,
            consensus_step=consensus_step, consensus_sel_size=15)
    ga.initialize_pop()

    if USE_EXPL:
        icm = ICM(input_shape=observation_space, output_size=action_space)

    # Run configs
    iterations = 10000
    log_step = 1
    NUM_EPISODE = Param(12, name='Num Episodes')

    # Checkpoints to resume training
    LOAD_CHECKPOINT = True
    CKPT_STEP = 5
    CHECKPOINT_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ckpt/%s/' % env_id)
    if not os.path.isdir(CHECKPOINT_FOLDER):
        os.mkdir(CHECKPOINT_FOLDER)

    if LOAD_CHECKPOINT and len([f for f in os.listdir(CHECKPOINT_FOLDER) if f.endswith('.pt')]) == pop_size:
        print('*** LOADING CHECKPOINT ***')
        for idx, indiv in enumerate(ga.pop):
            try:
                indiv.net.load_state_dict(torch.load(os.path.join(CHECKPOINT_FOLDER, 'indiv_%s.pt' % idx)))
            except Exception:
                indiv.net.load_state_dict(torch.load(os.path.join(CHECKPOINT_FOLDER, 'indiv_%s.pt' % idx),
                                                     map_location=torch.device('cpu')))
            indiv.weights_to_genes()

    memory = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1024. ** 3  # GB
    ray.init(num_cpus=os.cpu_count(), object_store_memory=4 * 1e+10 if memory > 45 else None)

    gpu_space = 1 / os.cpu_count() if torch.cuda.is_available() else 0
    @ray.remote(num_cpus=1, num_gpus=gpu_space)
    class ParallelEnv:
        def __init__(self, env_id):
            self.env = gym.make(env_id)
            self.env = apply_wrappers(self.env, env_id, FRAME_SKIP, FRAME_STACK)

            #if USE_EXPL:
            self.s = []
            self.s1 = []
            self.act = []
            self.step = 1000 #TODO#1
            self.step_idx = 0

        def simulate_episode(self, i, log, candidate, NUM_EPISODE):
            #if USE_EXPL:
            self.s = []
            self.s1 = []
            self.act = []

            r = 0
            for _ in range(NUM_EPISODE):
                state = self.env.reset()
                done = False
                while not done:
                    action, raw_action = candidate.predict(state)
                    next_state, reward, done, _ = self.env.step(action[0])
                    r += reward
                    #if USE_EXPL:
                    if self.step_idx % self.step == 0:
                        self.s.append(state)
                        self.s1.append(next_state)
                        self.act.append(raw_action)
                        self.step_idx = 0
                        if len(self.s) > 2048:
                            self.s = self.s[-2048:]
                            self.s1 = self.s1[-2048:]
                            self.act = self.act[-2048:]
                    self.step_idx += 1
                    state = next_state

            return r / NUM_EPISODE, (self.s, self.s1, self.act)

    envs = []
    for i in range(os.cpu_count()):
        envs.append(ParallelEnv.remote(env_id))
    actor_pool = ActorPool(envs)

    MAX_FIT = -np.inf
    # Iteration
    for i in range(iterations):
        log = False
        if i % log_step == 0:
            log = True

        tasks = actor_pool.map(lambda a, j: a.simulate_episode.remote(j, log, ga.pop[j], NUM_EPISODE.value),
                               range(pop_size))

        t = time.time()
        fitness, memories = zip(*tasks)
        print('Time to simulate the tasks', time.time() - t)

        if USE_EXPL:
            expl_fitness = []
            t = time.time()
            for s, s1, a in memories:  # Store memories inside icm module
                expl_fitness.append(icm.batch_reward(s[-768:], s1[-768:], a[-768:]).mean())
                icm.store_multiple(s, s1, a)
            print('Time to calculate icm reward', time.time() - t)

            top_expl = np.argpartition(fitness, -expl_elite)[-expl_elite:]

            t = time.time()
            fwd_loss, inv_loss = icm.train(2)
            print('Time to train icm', time.time() - t)

            if log:
                ga.logger.log_scalar(tag='Max Exploration Fitness', value=np.max(expl_fitness))
                ga.logger.log_scalar(tag='Mean Exploration Fitness', value=np.mean(expl_fitness))
                ga.logger.log_scalar(tag='STD Exploration fitness', value=np.std(expl_fitness))
                ga.logger.log_scalar(tag='Fwd Loss ICM', value=fwd_loss)
                ga.logger.log_scalar(tag='Inv Loss ICM', value=inv_loss)

        max_idx = np.argmax(fitness)
        if fitness[max_idx] > MAX_FIT and i > 0:  # Save checkpoint
            torch.save(ga.pop[max_idx].net.state_dict(), './checkpoint.pt')
            MAX_FIT = fitness[max_idx]

        if i % CKPT_STEP == 0 and i > 0:
            for idx, indiv in enumerate(ga.pop):
                torch.save(indiv.net.state_dict(), os.path.join(CHECKPOINT_FOLDER, 'indiv_%s.pt' % idx))

        if fitness[max_idx] >= finish_reward[env.spec.id]:
            print('****** Completed ******')
            sys.exit()

        t = time.time()
        maxx, mean, var = ga.simulate(fitness=fitness, preselected=top_expl,
                                      memories=memories, log=log)
        NUM_EPISODE.step()
        print('Num Episodes: %s' % NUM_EPISODE.value)
        print('Time to simulate ga', time.time() - t)

        # We manually call the update on weights
        for individual in ga.pop:
            individual.genes_to_weights()
