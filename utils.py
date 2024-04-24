import cv2
import gym
import gym_super_mario_bros
import numpy as np
from gym.wrappers import GrayScaleObservation
from nes_py.wrappers import JoypadSpace
# Import Vectorization Wrappers
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, SubprocVecEnv, vec_monitor
from stable_baselines3.common.monitor import Monitor
from typing import Callable
from stable_baselines3.common.utils import set_random_seed

MOVEMENT = [["right"], ["right", "A"]]
STAGE_PIXEL = 'SuperMarioBros-1-1-v0'
STAGE_RECTANGLE = 'SuperMarioBros-1-1-v3'

# This section analyzes the pre-processing that has been done to the environment. On the one hand, we have the SkipFrame function. By default, in each frame the game performs an action (a movement) and returns the reward for that action. What happens, is that to train the AI it is not necessary to make a move in each frame. That is why, the function executes the movement every X frames giving less work to do the training.
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

# By default the environment is composed of the RGB room. This data is unnecessary when training our model and we will get better results if we convert our game to a grayscale.
class ResizeEnv(gym.ObservationWrapper):
    def __init__(self, env, size):
        gym.ObservationWrapper.__init__(self, env)
        (oldh, oldw, oldc) = env.observation_space.shape
        newshape = (size, size, oldc)
        self.observation_space = gym.spaces.Box(low=0, high=255,
            shape=newshape, dtype=np.uint8)

    def observation(self, frame):
        height, width, _ = self.observation_space.shape
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        if frame.ndim == 2:
            frame = frame[:,:,None]
        return frame

class CustomRewardAndDoneEnv(gym.Wrapper):
    def __init__(self, env=None):
        super(CustomRewardAndDoneEnv, self).__init__(env)
        self.current_score = 0
        self.current_x = 0
        self.current_x_count = 0
        self.max_x = 0
    def reset(self, **kwargs):
        self.current_score = 0
        self.current_x = 0
        self.current_x_count = 0
        self.max_x = 0
        return self.env.reset(**kwargs)
    def step(self, action):
        state, reward, done, info = self.env.step(action)
        reward += max(0, info['x_pos'] - self.max_x)
        if (info['x_pos'] - self.current_x) == 0:
            self.current_x_count += 1
        else:
            self.current_x_count = 0
        if info["flag_get"]:
            reward += 500
            done = True
            print("GOAL")
        if info["life"] < 2:
            reward -= 500
            done = True
        self.current_score = info["score"]
        self.max_x = max(self.max_x, self.current_x)
        self.current_x = info["x_pos"]
        return state, reward / 10., done, info


def make_mario_env(env_id: str, rank: int, seed: int = 0) -> Callable:
    def _init() -> gym.Env:
        env = gym_super_mario_bros.make(env_id)
        env.reset(seed=seed + rank)
        env = Monitor(env)
        env = JoypadSpace(env, MOVEMENT)
        #env = CustomRewardAndDoneEnv(env)
        env = SkipFrame(env, skip=4)
        env = GrayScaleObservation(env, keep_dim=True)
        env = ResizeEnv(env, size=84)
        return env

    set_random_seed(seed)
    return _init

def make_parallel_env(env_id, n_envs, stack = True):
    env = SubprocVecEnv([make_mario_env(env_id, i) for i in range(n_envs)])
    if stack:
        env = VecFrameStack(env, 4, channels_order='last')
    return env

def make_single_env(env_id):
    env = make_mario_env(env_id, 0)()
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4, channels_order='last')
    return env
