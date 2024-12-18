
from game2048.agents import DQNAgentConfig
from torch import nn
config = DQNAgentConfig(
    epsilon_initial=0.9,
    epsilon_final=0.1,
    games_to_play=10000,
    invalid_move_penalty=-8.0,
    training_frequency=128,
    target_update_frequency_as_a_multiple_of_training=100,
    memory_capacity=10000,
    memory_batch_size=128,
    optim_lr=0.001,
    gamma=0.9
)


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(16, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
    
    def forward(self, x):
        return self.network(x)