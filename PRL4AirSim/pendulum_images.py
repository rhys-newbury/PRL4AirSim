import gymnasium as gym
import numpy as np

from stable_baselines3 import TD3
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)
from gymnasium.wrappers import PixelObservationWrapper, FrameStack
from gymnasium.wrappers import TransformObservation
import cv2

env = PixelObservationWrapper(
    gym.make("Pendulum-v1", render_mode="rgb_array", screen_dim=64), pixels_only=False
)
images = 100000
count = 0
while count < images:
    obs, _ = env.reset()
    done, truncated = False, False
    while not (done or truncated):
        action = env.action_space.sample()
        new_obs, reward, done, truncated, _ = env.step(action)
        cv2.imwrite(f"pend_images/{count}.png", new_obs["pixels"])
        count += 1
