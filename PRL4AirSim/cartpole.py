import gymnasium as gym
from TD3Trainer import TD3Trainer
from ReplayMemory import ReplayMemory
from gymnasium.wrappers import PixelObservationWrapper, FrameStack
import numpy as np
import torch
from VAE import VAE

env = PixelObservationWrapper(
    gym.make("Pendulum-v1", render_mode="rgb_array", screen_dim=64), pixels_only=False
)
obs, _ = env.reset()
# print(obs.keys())
print(obs["pixels"].shape)
print(env.action_space.shape)

vae = VAE(latent_dim=32)
vae.load_state_dict(torch.load("./VAE_32.pt"))
vae_ = vae.encoder
vae_.conv_output_dim = vae.conv_output_dim

for param in vae_.parameters():
    param.requires_grad = False


t = TD3Trainer(
    image_input_dims=(
        obs["pixels"].shape[2],
        obs["pixels"].shape[0],
        obs["pixels"].shape[1],
    ),
    n_actions=env.action_space.shape[0],
    replayMemory_size=10000,
    batch_size=1024,
    max_action=2.0,
    policy_noise=0.02,
    noise_clip=0.005,
    image_extractor=lambda x: vae_,
    learningRate=0.0005,
)


for epoch in range(10000):
    m = ReplayMemory(100000)
    # ("state", "action", "next_state", "reward", "not_done")
    total_reward = []
    for _ in range(10):
        obs, _ = env.reset()

        obs = {
            b"image": obs["pixels"].astype(np.float32),
            b"velocity": obs["state"],
        }

        total_reward.append(0)
        done, truncated = False, False
        while not (done or truncated):
            action = t.choose_action(obs).cpu().detach().float().numpy()[0]
            new_obs, reward, done, truncated, _ = env.step(action)
            new_obs = {
                b"image": new_obs["pixels"].astype(np.float32),
                b"velocity": new_obs["state"],
            }
            total_reward[-1] += reward
            m.push(obs, action, new_obs, reward, not done)
            obs = new_obs

    for _ in range(2):  # Train the agent for 10 steps
        print("learning...")
        data = m.sample(128)

        t.learn(data)

    print(f"Epoch: {epoch+1}, Total Reward: {np.mean(total_reward)}")
