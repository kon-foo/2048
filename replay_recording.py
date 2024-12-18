import argparse
import time

from game2048.recording import Player
from game2048.visualize import ConsoleVisualizer

def main():
    parser = argparse.ArgumentParser(description='Replay a recording of a game of 2048.')
    parser.add_argument('file', type=str, help='Path to the recording file.')
    parser.add_argument('--speed', type=float, default=3, help='Speed of the replay in moves per second.')
    args = parser.parse_args()

    visualizer = ConsoleVisualizer()
    player = Player(visualizer, speed=args.speed)
    player.load_file(args.file)
    player.play()


if __name__ == "__main__":
    main()