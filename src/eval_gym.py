import gym
import torch
import time
import numpy as np
from ga_icm import apply_wrappers
from model_old import Net
torch.set_grad_enabled(False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('**** PYTORCH DEVICE: %s ****' % device)

# Environment
env_id = 'CarRacing-v0'
FRAME_SKIP = 2  # 1 for disabled
FRAME_STACK = 1
env = gym.make(env_id)
env = apply_wrappers(env, env_id, FRAME_SKIP, FRAME_STACK)
env._max_episode_steps = 100000

action_space = env.action_space
observation_space = env.observation_space

# Calculate gene_size
net = Net(input_shape=observation_space, output_size=action_space, channels=12, conv_channels=16)
# net = Net(input_shape=observation_space, output_size=action_space)
net.load_state_dict(torch.load('./carRacing_nostack_12c16conv.pt', map_location=torch.device('cpu')))
net.eval()

while True:
    state = env.reset()
    done = False
    r = 0
    while not done:
        if not isinstance(state, torch.Tensor):
            if state.ndim == 3:
                state = torch.tensor(state.copy(), dtype=torch.float32).permute(2, 0, 1)
            else:
                state = torch.tensor(state, dtype=torch.float32)

        action = net(state[None])
        if net.is_discrete:
            action = np.argmax(action, axis=1)

        state, reward, done, _ = env.step(action[0])
        r += reward
        env.render()
        time.sleep(0.01)
    print('Reward:', r)
