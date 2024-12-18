import argparse
from game2048.agents import InferenceAgent, RandomAgent, VerticalHorizontalVertical
from game2048.visualize import ConsoleVisualizer

from train import DQN



def main():
    parser = argparse.ArgumentParser(description='Replay a recording of a game of 2048.')
    parser.add_argument('file', type=str, help='Path to the recording file.')
    parser.add_argument('--speed', type=float, default=3, help='Speed of the replay in moves per second.')
    parser.add_argument('--compare', action='store_true', help='Compare the agent to a random agent playing the same seed.')
    parser.add_argument('--seed', type=int, help='Seed for the game.')
    parser.add_argument('--cheat', action='store_false', help='If True, the agent will self-unstuck and avoid making the same ivalid move twice.')
    args = parser.parse_args()

    agent = InferenceAgent(
        self_unstuck=args.cheat,
        model_path=args.file, 
        network_class=DQN, 
        moves_per_second=args.speed, 
        visualizer=ConsoleVisualizer())
    agent.play(seed=args.seed)

    if args.compare:
        seed = agent.game.seed
        random_agent = RandomAgent()
        result = random_agent.play(seed=seed)
        print(f"Random agent score: {result.score}")

        upleft_agent = VerticalHorizontalVertical()
        result = upleft_agent.play(seed=seed)
        print(f"Up-Left agent score: {result.score}")




if __name__ == "__main__":
    main()