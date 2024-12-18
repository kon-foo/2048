# src/game2048/visualization/console.py
from typing import Optional, Dict, Any
import os
from ..engine import GameState, Move

from .base_visualizer import Visualizer

class ConsoleVisualizer(Visualizer):
    """Simple console-based visualizer for 2048 game"""
    
    # ANSI color codes for different tile values
    COLORS = {
        0: ('\033[0m', '\033[0m'),      # Default
        2: ('\033[97m', '\033[0m'),     # White
        4: ('\033[93m', '\033[0m'),     # Yellow
        8: ('\033[91m', '\033[0m'),     # Red
        16: ('\033[31m', '\033[0m'),    # Dark Red
        32: ('\033[95m', '\033[0m'),    # Magenta
        64: ('\033[94m', '\033[0m'),    # Blue
        128: ('\033[96m', '\033[0m'),   # Cyan
        256: ('\033[92m', '\033[0m'),   # Green
        512: ('\033[32m', '\033[0m'),   # Dark Green
        1024: ('\033[33m', '\033[0m'),  # Dark Yellow
        2048: ('\033[35m', '\033[0m'),  # Dark Magenta
        4096: ('\033[34m', '\033[0m'),  # Dark Blue
        8192: ('\033[36m', '\033[0m'),  # Dark Cyan
    }
    
    def __init__(self):
        pass
    
    def clear_screen(self):
        """Clear the console screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def _get_cell_str(self, value: int) -> str:
        """Format a single cell with proper spacing and color"""
        if value == 0:
            return "    ·"  # Centered dot for empty cells
        
        # Get colors, defaulting to no color if value not in COLORS
        color_start, color_end = self.COLORS.get(value, ('', ''))
        return f"{color_start}{value:>5}{color_end}"
    
    def render(self, state: GameState, metadata: dict[str, any] = {} ) -> None:
        """Render the current game state to the console"""
        self.clear_screen()
        
        # Print header with game info
        print("\n" + "=" * 40)
        print(f"Move: {state.move_count:<5} Score: {state.score:<10}")
        if state.last_move is not None:
            print(f"Last Move: {state.last_move.name}")
            print(f"Invalid Moves: {state.invalid_move_count}")
        # Print metadata if available
        for key, value in metadata.items():
            # Truncate long values
            str_value = str(value)
            if len(str_value) > 50:
                str_value = str_value[:47] + "..."
            print(f"  {key}: {str_value}")            
        print("=" * 40 + "\n")
        
        # Print the board
        for row in state.board:
            # Print cells with borders
            print("│", end=" ")
            for cell in row:
                print(self._get_cell_str(cell), end=" │ ")
            print("\n" + "─" * (7 * len(row) + 5))
        

        
        print("\nGame Over!" if state.game_over else "")

# Example usage:
if __name__ == "__main__":
    # Test visualization
    visualizer = ConsoleVisualizer()
    test_state = GameState(
        board=[
            [2, 4, 8, 16],
            [32, 64, 128, 256],
            [512, 1024, 2048, 4],
            [2, 4, 8, 0]
        ],
        score=12345,
        game_over=False
    )
    
    visualizer.render(
        test_state,
        move=Move.UP,
        metadata={"strategy": "corner", "evaluation": 0.756}
    )