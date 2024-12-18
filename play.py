from game2048.agents import ManualAgent, RandomAgent
from game2048.visualize import ConsoleVisualizer

agent = ManualAgent()
agent.play(save_recording=True)

# agent = RandomAgent(
#     visualizer=ConsoleVisualizer()
# )
# agent.play(save_recording=True)