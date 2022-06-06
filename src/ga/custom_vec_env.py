import numpy as np
import gym
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.vec_env.util import copy_obs_dict, dict_to_obs, obs_space_info
from baselines.common.tile_images import tile_images


class DummyListEnvs(VecEnv):
    def __init__(self, env_name, num_envs):
        """
        Arguments:

        env_fns: iterable of callables      functions that build environments
        """
        self.envs = [gym.make(env_name) for _ in range(num_envs)]
        env = self.envs[0]
        VecEnv.__init__(self, num_envs, env.observation_space, env.action_space)
        obs_space = env.observation_space
        self.keys, shapes, dtypes = obs_space_info(obs_space)

        self.buf_obs = { k: np.zeros((self.num_envs,) + tuple(shapes[k]), dtype=dtypes[k]) for k in self.keys }
        self.spec = self.envs[0].spec

    def step_async(self, actions):
        raise NotImplementedError

    def step_wait(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def close(self):
        for env in self.envs:
            env.close()

    def get_attr(self, attr_name, indices=None):
        """Return attribute from vectorized environment (see base class)."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, attr_name) for env_i in target_envs]

    def set_attr(self, attr_name, value, indices=None):
        """Set attribute inside vectorized environments (see base class)."""
        target_envs = self._get_target_envs(indices)
        for env_i in target_envs:
            setattr(env_i, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Call instance methods of vectorized environments."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, method_name)(*method_args, **method_kwargs) for env_i in target_envs]

    def get_images(self):
        return [env.render(mode='rgb_array') for env in self.envs]

    def _save_obs(self, e, obs):
        for k in self.keys:
            if k is None:
                self.buf_obs[k][e] = obs
            else:
                self.buf_obs[k][e] = obs[k]

    def _obs_from_buf(self):
        return dict_to_obs(copy_obs_dict(self.buf_obs))

    def render(self, mode='human'):
        imgs = self.get_images()
        bigimg = tile_images(imgs)
        if mode == 'human':
            self.get_viewer().imshow(bigimg)
            return self.get_viewer().isopen
        elif mode == 'rgb_array':
            return bigimg
        else:
            raise NotImplementedError
