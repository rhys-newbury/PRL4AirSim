import copy

# import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import ReplayMemory


# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477
class ImageHead(nn.Module):
    def __init__(self, image_input_dims: tuple):
        super().__init__()
        self.image_input_dims = image_input_dims
        self.maxpooling = nn.MaxPool2d((2, 2), stride=2)

        self.image_conv1 = nn.Conv2d(
            image_input_dims[0], 16, kernel_size=(6, 6), stride=(2, 2)
        )
        self.image_conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1))

        self.conv_output_dim = self.calculate_conv_output_dims()

    def calculate_conv_output_dims(self):
        state = torch.zeros(1, *self.image_input_dims).float()
        print("inpute state :", state.size())

        x = self.maxpooling(F.relu(self.image_conv1(state)))
        print("layer 1", x.size())
        x = self.maxpooling(F.relu(self.image_conv2(x)))
        print("layer 2", x.size())

        return int(np.prod(x.size()))

    def forward(self, image):
        image = self.maxpooling(F.relu(self.image_conv1(image)))
        image = self.maxpooling(F.relu(self.image_conv2(image)))
        return image.view(image.size()[0], -1)


class Actor(nn.Module):
    def __init__(
        self,
        image_input_dims: tuple,
        action_dim: int,
        max_action: float,
        image_extractor=ImageHead,
    ):
        super().__init__()

        self.image_head = image_extractor(image_input_dims)
        self.vel_fc1 = nn.Linear(3, 16)

        self.out_fc1 = nn.Linear(self.image_head.conv_output_dim + 16, 16)
        self.out_fc2 = nn.Linear(16, action_dim)

        self.max_action = max_action

    def forward(self, image: torch.tensor, velocity: torch.tensor):
        # x = image[:, 0, :, :].view(image.shape[0], 4096)
        image = self.image_head(image)

        velocity = F.relu(self.vel_fc1(velocity))

        concatinated_tensor = torch.cat((image, velocity), 1)

        x = F.relu(self.out_fc1(concatinated_tensor))
        x = self.out_fc2(x)
        return torch.tanh(x) * self.max_action


class Critic(nn.Module):
    def __init__(
        self, image_input_dims: tuple, action_dim: int, image_extractor=ImageHead
    ):
        super().__init__()

        # Q1 architecture
        self.image_head_1 = image_extractor(image_input_dims)
        self.vel_fc_1 = nn.Linear(3, 16)
        self.l1_1 = nn.Linear(self.image_head_1.conv_output_dim + 16 + action_dim, 16)
        self.l2_1 = nn.Linear(16, 1)

        # Q1 architecture
        self.image_head_2 = image_extractor(image_input_dims)
        self.vel_fc_2 = nn.Linear(3, 16)
        self.l1_2 = nn.Linear(self.image_head_2.conv_output_dim + 16 + action_dim, 16)
        self.l2_2 = nn.Linear(16, 1)

    def forward(self, image, velocity, action):

        # x = image[:, 0, :, :].view(image.shape[0], 4096)

        q1 = self.image_head_1(image)
        v1 = self.vel_fc_1(velocity)
        q1 = torch.cat((q1, v1, action), 1)
        q1 = self.l1_1(q1)
        q1 = self.l2_1(q1)

        q2 = self.image_head_2(image)
        v2 = self.vel_fc_2(velocity)
        q2 = torch.cat((q2, v2, action), 1)
        q2 = self.l1_2(q2)
        q2 = self.l2_2(q2)

        return q1, q2

    def Q1(self, image, velocity, action):

        # x = image[:, 0, :, :].view(image.shape[0], 4096)

        q1 = self.image_head_1(image)
        v1 = self.vel_fc_1(velocity)
        q1 = torch.cat((q1, v1, action), 1)
        q1 = self.l1_1(q1)
        q1 = self.l2_1(q1)

        return q1


class TD3Trainer:
    def __init__(
        self,
        image_input_dims: np.array,
        n_actions: int,
        replayMemory_size: int,
        batch_size: int,
        learningRate: float = 0.01,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        replace_target_count_episode: int = 100,
        save_model_count_episode: int = 250,
        checkpoint_episode: int = 250,
        checkpoint_file: str = "model_saves/dqn",
        number_dimensions: int = 2,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        max_action=0.25,
        image_extractor=ImageHead,
    ):
        self.image_input_dims = image_input_dims
        self.n_actions = n_actions
        self.device = device
        self.actor = Actor(image_input_dims, n_actions, max_action, image_extractor).to(
            device
        )
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=learningRate
        )

        self.critic = Critic(image_input_dims, n_actions, image_extractor).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=learningRate
        )

        self.max_action = max_action
        self.discount = discount_factor
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.batch_size = batch_size
        self.memory = ReplayMemory.ReplayMemory(replayMemory_size)
        self.replayMemory_size = replayMemory_size
        self.checkpoint_episode = checkpoint_episode
        self.checkpoint_file = checkpoint_file
        self.epsilon = epsilon
        self.save_model_count_episode = save_model_count_episode
        self.replace_target_count_episode = replace_target_count_episode
        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def choose_action(self, observation: dict):
        if np.random.random() > self.epsilon:
            return (
                (
                    (
                        torch.tensor(
                            [np.random.random() for _ in range(self.n_actions)]
                        )
                        * 2
                        - 1
                    )
                    * self.max_action
                )
                .to(device)
                .unsqueeze(axis=0)
            )
        
        print(observation)
        
        
        image = torch.tensor(
            np.reshape(np.array(observation["image"]), (1, *self.image_input_dims)),
            dtype=torch.float,
        ).to(device)
        velocity = torch.tensor(
            np.array(observation["velocity"]).reshape((1, 3)), dtype=torch.float
        ).to(device)

        action = self.actor.forward(image, velocity)

        return action

    def learn(self, transitions):
        self.total_it += 1

        # self.network.optimizer.zero_grad()
        self.memory.pushCounter += 1

        # if self.memory.pushCounter % self.replace_target_count_episode == 0:
        #     print(
        #         "Transfer weights to target network at step {}".format(
        #             self.memory.pushCounter
        #         )
        #     )
        #     self.target_network.load_state_dict(self.network.state_dict())

        batch = ReplayMemory.Transition(*zip(*transitions))

        state = (
            torch.tensor(
                np.array(
                    [i[b"image"].reshape(*self.image_input_dims) for i in batch.state]
                )
            )
            .to(device)
            .float(),
            torch.tensor(np.array([i[b"velocity"] for i in batch.state]))
            .to(device)
            .float(),
        )

        next_state = (
            torch.tensor(
                np.array(
                    [
                        i[b"image"].reshape(*self.image_input_dims)
                        for i in batch.next_state
                    ]
                )
            )
            .to(device)
            .float(),
            torch.tensor(np.array([i[b"velocity"] for i in batch.next_state]))
            .to(device)
            .float(),
        )

        action = torch.tensor(batch.action).to(device)
        reward = torch.tensor(batch.reward).to(device).unsqueeze(dim=-1).float()
        not_done = torch.tensor(batch.not_done).to(device).unsqueeze(dim=-1)

        indices = np.arange(self.batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )

            next_action = (self.actor_target(*next_state) + noise).clamp(
                -self.max_action, self.max_action
            )

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(*next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(*state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor losse
            actor_loss = -self.critic.Q1(*state, self.actor(*state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

            for param, target_param in zip(
                self.actor.parameters(), self.actor_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(
            torch.load(filename + "_critic_optimizer")
        )
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)

    def decrement_epsilon(self):
        # if self.memory.pushCounter < self.replayMemory_size and self.memory.pushCounter > self.replayMemory_size * 0.2 * 0.99:
        if self.memory.pushCounter > self.replayMemory_size:
            self.epsilon = max(
                0,
                1.0
                - (
                    (self.memory.pushCounter - self.replayMemory_size)
                    / self.replayMemory_size
                ),
            )

    def test(self):
        print("Testing network")
        image = torch.zeros(1, *self.image_input_dims).float().to(self.device)
        velocity = torch.zeros((1, 3)).float().to(self.device)
        print(
            "Input shapes: [image]: {} [velocity]: {}".format(
                image.size(), velocity.size()
            )
        )
        output = self.actor.forward(image, velocity)
        print("Output: {}".format(output))
        q1, q2 = self.critic.forward(image, velocity, output)
        print(f"q1={q1}, q2={q2}")


if __name__ == "__main__":
    print("test")
    model = TD3Trainer(
        learningRate=0.001,
        n_actions=2,
        image_input_dims=(2, 64, 64),
        replayMemory_size=100,
        batch_size=2,
    )
    print("total parameters: ", sum(p.numel() for p in model.actor.parameters()))
    print(
        "total trainable parameters: ",
        sum(p.numel() for p in model.actor.parameters() if p.requires_grad),
    )
    print("total data points: ", (10 * 32 * 5000) / 30)
    model.test()
