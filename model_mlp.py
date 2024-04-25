import gym
import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class MarioNetMLP1(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box, features_dim):
        super(MarioNetMLP1, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0] * observation_space.shape[1] * observation_space.shape[2]
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_input_channels, 1764),
            nn.ReLU(),
            nn.Linear(1764, 1764),
            nn.ReLU(),
            nn.Linear(1764, 1764),
            nn.ReLU(),
            nn.Linear(1764, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.mlp(observations)

class MarioNetMLP2(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box, features_dim):
        super(MarioNetMLP2, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0] * observation_space.shape[1] * observation_space.shape[2]
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_input_channels, 1764),
            nn.ReLU(),
            nn.Linear(1764, 882),
            nn.ReLU(),
            nn.Linear(882, 441),
            nn.ReLU(),
            nn.Linear(441, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.mlp(observations)

class MarioNetMLP3(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box, features_dim):
        super(MarioNetMLP3, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0] * observation_space.shape[1] * observation_space.shape[2]
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_input_channels, 882),
            nn.ReLU(),
            nn.Linear(882, 441),
            nn.ReLU(),
            nn.Linear(441, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.mlp(observations)

class MarioNetMLP4(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box, features_dim):
        super(MarioNetMLP4, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0] * observation_space.shape[1] * observation_space.shape[2]
        self.start = nn.Sequential(
            nn.Flatten(),
        )
        self.first_1 = nn.Sequential(
            nn.Linear(n_input_channels / 4, 220),
            nn.ReLU(),
        )
        self.first_2 = nn.Sequential(
            nn.Linear(n_input_channels / 4, 220),
            nn.ReLU(),
        )
        self.first_3 = nn.Sequential(
            nn.Linear(n_input_channels / 4, 220),
            nn.ReLU(),
        )
        self.first_4 = nn.Sequential(
            nn.Linear(n_input_channels / 4, 220),
            nn.ReLU(),
        )
        self.second_1 = nn.Sequential(
            nn.Linear(440, 220),
            nn.ReLU(),
        )
        self.second_2 = nn.Sequential(
            nn.Linear(440, 220),
            nn.ReLU(),
        )
        self.third = nn.Sequential(
            nn.Linear(440, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        result = self.start(observations)
        result_first_1 = self.first_1(result[:, :int(observations.shape[1] / 4)])
        result_first_2 = self.first_2(result[:, int(observations.shape[1] / 4):int(observations.shape[1] / 2)])
        result_first_3 = self.first_3(result[:, int(observations.shape[1] / 2):int(3 * observations.shape[1] / 4)])
        result_first_4 = self.first_4(result[:, int(3 * observations.shape[1] / 4):])
        result_second_1 = self.second_1(th.cat((result_first_1, result_first_2), 1))
        result_second_2 = self.second_2(th.cat((result_first_3, result_first_4), 1))
        result_third = self.third(th.cat((result_second_1, result_second_2), 1))
        return result_third

