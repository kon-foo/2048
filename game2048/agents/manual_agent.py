# src/game2048/agents/human.py
from typing import Optional, Dict
from ..engine import Game2048, Move
from .base_agent import Agent
from game2048.utils import get_arrow_key
from game2048.visualize import Visualizer, ConsoleVisualizer

class ManualAgent(Agent):
    """
    Agent that allows human players to control the game via keyboard input.
    Supports both arrow keys and WASD controls.
    """
    # Default key mappings
    KEY_MAPPINGS = {
        'w': Move.UP,
        'a': Move.LEFT,
        's': Move.DOWN,
        'd': Move.RIGHT,
        'up': Move.UP,
        'left': Move.LEFT,
        'down': Move.DOWN,
        'right': Move.RIGHT
    }
    
    def __init__(self, **kwargs):
        if 'visualizer' not in kwargs:
            kwargs['visualizer'] = ConsoleVisualizer()
        super().__init__(**kwargs)
    
    def _print_controls(self) -> None:
        """Print the control scheme once"""
        print("\nControls:")
        print("  ↑  or W = Up")
        print("  ↓  or S = Down")
        print("  ←  or A = Left")
        print("  →  or D = Right")
        print("  Q = Quit")
        print("\n")
        input("Press any key to start...")
        self._printed_controls = True

    def play(self, save_recording = False, max_rounds = 0):
        self._print_controls()
        return super().play(save_recording, max_rounds)
    
    def get_move(self) -> Optional[Move]:
        """Get the next move from human input"""
        while True:
            key = get_arrow_key()
            # Handle quit
            if key == 'q':
                print("\nGame terminated by player.")
                raise KeyboardInterrupt
            
            # Try to get move from key mapping
            move = self.KEY_MAPPINGS.get(key)
            if move is None:
                print(f"Invalid key: {key}. Please use arrow keys or WASD of Q to quit.")
                continue
            else:
                return move
            
