import json
import time
from .game_record import GameRecord
from ..engine.engine import GameState, Move, Game2048
from ..visualize import Visualizer
from ..utils import get_arrow_key

class Player:
    def __init__(self, visualizer: Visualizer, speed: float = 0.5):
        """
        Args:
            visualizer: the visualizer to render the game state
            speed: the speed of the replay in moves per second
        """
        self.record: GameRecord = None
        self.visualizer: Visualizer = visualizer
        self.seconds_per_move = 1 / speed if speed != 0 else 0
        self.manual = True if speed == 0 else False
        self.all_states: list[GameState] = []
        self.current_index: int = 0  # Tracks the current state index
    
    def load(self, record: GameRecord):
        self.record = record

    def load_file(self, file_path: str):
        with open(file_path, "r") as file:
            data_raw = json.load(file)
            data = GameRecord.model_validate(data_raw)
        self.load(data)

    def _build_state_history(self):
        """Replay all moves and store states in self.all_states."""
        game = Game2048(size=self.record.size, seed=self.record.seed)
        self.all_states = [game.get_current_state()]  # Initial state
        for move in self.record.last_state.performed_moves:
            game.make_move(move)
            self.all_states.append(game.get_current_state())

    def play(self):
        if self.record is None:
            raise ValueError("No record loaded")
        
        self._build_state_history()
        self.current_index = 0  # Reset state pointer

        while True:
            current_state = self.all_states[self.current_index]
            self.visualizer.render(current_state)

            # Handle user input if manual mode is enabled
            if self.manual:
                user_input = get_arrow_key()
                if user_input == "right":
                    if self.current_index < len(self.all_states) - 1:
                        self.current_index += 1
                    else:
                        print("Already at the last state.")
                elif user_input == "left":
                    if self.current_index > 0:
                        self.current_index -= 1
                    else:
                        print("Already at the first state.")
                elif user_input == "q":
                    print("Exiting replay.")
                    break
                else:
                    print("Invalid input. Use 'left', 'right', or 'q'.")
            else:
                time.sleep(self.seconds_per_move)
                if self.current_index < len(self.all_states) - 1:
                    self.current_index += 1
                else:
                    print("Replay complete.")
                    break