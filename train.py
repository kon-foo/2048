from torch import nn

from game2048.agents import RandomAgent, DQNAgent, DQNAgentConfig, InferenceAgent
from game2048.visualize import ConsoleVisualizer
from .train_base_config import config, DQN

if __name__ == "__main__":
    agent = DQNAgent(config, network_class=DQN)
    agent.play_many()
    agent.save(filename=f"{agent.name}_final.pt")

