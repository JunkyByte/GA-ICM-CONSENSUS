import ray
import torch
from torch import nn
from gym.spaces import Box, Discrete
from torch.nn.functional import mse_loss, cross_entropy
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ICM_Model(nn.Module):
    def __init__(self, input_shape, output_size, channels=32, conv_channels=32):
        super(ICM_Model, self).__init__()

        if len(input_shape.shape) == 1:  # One dimensional observation space
            self.is_image = False
            self.input_shape = input_shape.shape
        elif len(input_shape.shape) == 3:
            self.is_image = True
            self.rescale = True if input_shape.high.any() == 255 else False
            self.input_shape = tuple(input_shape.shape[i] for i in [2, 0, 1])

        if isinstance(output_size, Discrete):
            self.output_size = output_size.n
            self.is_discrete = True
        elif isinstance(output_size, Box):
            self.out_bounds = torch.tensor(np.array([list(output_size.low), list(output_size.high)]), dtype=torch.float).to(device)
            self.output_size = output_size.shape[0]
            self.is_discrete = False

        # Projection
        if self.is_image:
            self.phi = nn.Sequential(
                nn.Conv2d(self.input_shape[0], conv_channels, kernel_size=3, stride=2, padding=1),
                nn.ELU(),
                nn.Conv2d(conv_channels, conv_channels, kernel_size=3, stride=2, padding=1),
                nn.ELU(),
                nn.Conv2d(conv_channels, conv_channels, kernel_size=3, stride=2, padding=1),
                nn.ELU(),
                nn.Conv2d(conv_channels, conv_channels, kernel_size=3, stride=2, padding=1),
                nn.ELU()
            )
        else:
            self.phi = nn.Sequential(
                nn.Linear(self.input_shape[0], channels),
                nn.ReLU(),
                nn.Linear(channels, channels),
                nn.ReLU(),
                nn.Linear(channels, channels),
                nn.ReLU()
            )

        self.phi_shape = self.phi(torch.randn(*((1,) + tuple(self.input_shape)))).view(-1).size().numel()

        # Forward Model
        self.fwd = nn.Sequential(
            nn.Linear(self.phi_shape + self.output_size, channels),
            nn.ReLU(),
            nn.Linear(channels, self.phi_shape)
        )

        inv_modules = [nn.Linear(self.phi_shape * 2, channels),
                       nn.ELU(),
                       nn.Linear(channels, self.output_size)]

        if not self.is_discrete:
            inv_modules.append(nn.Sigmoid())

        # Inverse Model
        self.inv = nn.Sequential(*inv_modules)

    def forward(self, *input):
        obs, action = input
        if self.is_discrete:
            action = torch.tensor(action, dtype=torch.long).squeeze(1)
            action = torch.eye(self.output_size)[action].to(device)
        else:
            action = torch.tensor(action, dtype=torch.float).to(device).squeeze(1)

        phi = self.phi_state(obs)
        x = torch.cat((phi, action.float()), -1)
        phi_hat = self.fwd(x)
        return phi_hat

    def phi_state(self, s):
        if self.is_image:
            x = s.view(-1, *self.input_shape)
            if self.rescale:
                x = x / 255.
        else:
            x = s.view(-1, self.input_shape[0])

        x = self.phi(x)
        return x.view(x.size(0), -1)

    def inverse_pred(self, s, s1):
        s = self.phi_state(s.float())
        s1 = self.phi_state(s1.float())
        x = torch.cat((s, s1), -1)
        a_hat = self.inv(x)
        if self.is_discrete:
            return a_hat

        return a_hat * (self.out_bounds[1] - self.out_bounds[0]) + self.out_bounds[0]

    def curiosity_rew(self, s, s1, a):
        s = torch.tensor(np.array(s), dtype=torch.float, requires_grad=False, device=device)
        s1 = torch.tensor(np.array(s1), dtype=torch.float, requires_grad=False, device=device)
        phi_hat = self.forward(s, a)
        phi_s1 = self.phi_state(s1)
        cur_rew = 1 / 2 * (torch.norm(phi_hat - phi_s1, p=2, dim=-1) ** 2)
        return cur_rew


class Memory:
    def __init__(self, max_memory):
        self.max_memory = max_memory
        self.state = []
        self.new_state = []
        self.action = []
        self.idx = 0

    def __len__(self):
        return len(self.state)

    def store_multiple(self, si, si1, ai):
        self.state.extend(si)
        self.new_state.extend(si1)
        self.action.extend(ai)
        if len(self.state) > self.max_memory:
            self.state = self.state[-self.max_memory:]
            self.new_state = self.new_state[-self.max_memory:]
            self.action = self.action[-self.max_memory:]

    def store_transition(self, s, s1, a):
        if len(self.state) <= self.max_memory:
            self.state.append(s)
            self.new_state.append(s1)
            self.action.append(a)
        else:
            self.state[self.idx] = s
            self.new_state[self.idx] = s1
            self.action[self.idx] = a
            self.idx = (self.idx + 1) % self.max_memory
        assert len(self.state) == len(self.new_state) == len(self.action)

    def clear_memory(self):
        del self.state[:]
        del self.new_state[:]
        del self.action[:]

    def sample(self, bs):
        idx = np.random.randint(len(self.state), size=bs)
        state, new_state, action = [], [], []
        for i in idx:
            state.append(self.state[i])
            new_state.append(self.new_state[i])
            action.append(self.action[i])
        return state, new_state, action

    def __iter__(self):
        return zip(self.state, self.new_state, self.action)

    def update(self, **kwargs):
        pass


class ICM:
    def __init__(self, input_shape, output_size, channels=32, conv_channels=32):
        self.icm = ICM_Model(input_shape, output_size, channels, conv_channels).to(device)
        self.icm_opt = torch.optim.Adam(self.icm.parameters(), lr=1e-3)
        self.memory = Memory(1000)
        self.beta = 0.2
        self.bs = 128

    def store_memory(self, memory):
        for s, s1, a in memory:
            self.store_transition(s, s1, a)

    def store_multiple(self, s, s1, a):
        self.memory.store_multiple(s, s1, a)

    def store_transition(self, s, s1, a):
        self.memory.store_transition(s, s1, a)

    def memory_reward(self, memory):
        return self.icm.curiosity_rew(memory.state, memory.new_state, memory.action).mean().cpu().detach().numpy()

    def batch_reward(self, state, new_state, action):
        return self.icm.curiosity_rew(state, new_state, action).cpu().detach().numpy()

    def train(self, iters):
        fwd_loss, inv_loss = 0, 0
        for i in range(iters):
            f_loss, i_loss = self._train()
            fwd_loss += f_loss
            inv_loss += i_loss
        return fwd_loss / iters, inv_loss / iters

    def _train(self):
        state, new_state, action = self.memory.sample(self.bs)

        state = torch.tensor(np.array(state), dtype=torch.float).detach().to(device)
        new_state = torch.tensor(np.array(new_state), dtype=torch.float).detach().to(device)
        action = torch.tensor(np.array(action), dtype=torch.float).detach().to(device)

        phi_hat = self.icm.forward(state, action)
        phi_true = self.icm.phi_state(new_state)
        fwd_loss = mse_loss(input=phi_hat, target=phi_true.detach(), reduction='mean').mean(-1)
        a_hat = self.icm.inverse_pred(state, new_state)

        if self.icm.is_discrete:
            inv_loss = cross_entropy(input=a_hat, target=action.long().squeeze(1), reduction='mean')
        else:
            inv_loss = mse_loss(input=a_hat, target=action.squeeze(1), reduction='mean')

        # Calculate joint loss
        inv_loss = (1 - self.beta) * inv_loss
        fwd_loss = self.beta * fwd_loss  # * np.prod(self.icm.phi_shape)
        loss = inv_loss + fwd_loss

        self.icm_opt.zero_grad()
        loss.backward()
        self.icm_opt.step()

        return fwd_loss.cpu().detach().numpy(), inv_loss.cpu().detach().numpy()
