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
from VAEExtractor import VAEExtractor

env = PixelObservationWrapper(
    gym.make("Pendulum-v1", render_mode="rgb_array", screen_dim=64), pixels_only=False
)

print(env.observation_space)
# The noise objects for TD3
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(
    mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
)

model = TD3(
    "MultiInputPolicy",
    env,
    action_noise=action_noise,
    verbose=1,
    policy_kwargs={"features_extractor_class": VAEExtractor},
)
# model.learn(total_timesteps=100000, log_interval=10)
# model.save("td3_pendulum")
vec_env = model.get_env()

del model  # remove to demonstrate saving and loading

model = TD3.load("td3_pendulum")
print("running......")
obs = vec_env.reset()
for _ in range(100):
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")
        if dones:
            break
    input("enter to continue")
