from abc import ABC, abstractmethod
from torch import nn
import torch


class RLNetwork(nn.Module, ABC):
    def __init__(self, learning_rate: float, num_actions: int, image_input_dims: tuple):
        super().__init__()
        self.learning_rate = learning_rate
        self.num_actions = num_actions
        self.image_input_dims = image_input_dims
        self.device = torch.device("cuda:0")

    @abstractmethod
    def forward(self, image: torch.tensor, velocity: torch.tensor):
        pass
