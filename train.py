from torch import nn

from game2048.agents import RandomAgent, DQNAgent, DQNAgentConfig, InferenceAgent
from game2048.visualize import ConsoleVisualizer

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

if __name__ == "__main__":

    config = DQNAgentConfig(
        games_to_play=10000,
        epsilon_initial=0.9,
        invalid_move_penalty=-32.0,
        training_frequency=128,
        memory_batch_size=128,
        optim_lr=0.001,
        target_update_frequency_as_a_multiple_of_training=100,
        gamma=0.99
        )
    agent = DQNAgent(config, network_class=DQN)
    agent.play_many()
    agent.save(filename=f"{agent.name}_final.pt")

