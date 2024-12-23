import csv
import os
from pathlib import Path
from abc import ABC, abstractmethod

from game2048 import Game2048, Move, GameRecord
from game2048.visualize import Visualizer
from .base_agent_metrics import AgentMetrics


class Agent(ABC):
    """
    An Agent can have a lifetime of multiple games.
    An Agent runs a single game at a time.
    It calls its own _get_move method to get the next move and then calls makes the move in the game.
    Base interface for all agents.
    """

    def __init__(self, save_base_path: str = "results", visualizer: Visualizer = None):
        """
        Initialize the agent.
        """

        self.game: Game2048 = None
        self.save_base_path = save_base_path
        self.save_recordings_path = save_base_path + "/recordings"
        self.save_metrics_path = save_base_path + "/metrics"
        self.move_metadata: dict[int, dict[str, any]] = {}
        self.agent_metadata: dict[str, any] = {}
        self.metrics = AgentMetrics()
        self.visualizer: Visualizer = visualizer


    @property
    def name(self) -> str:
        """
        Return the name of the agent.
        Should be overriden by the subclass.
        Is used to save the results of the agent.
        """
        return self.__class__.__name__
    

    def add_move_metadata(self, move_idx: int = None, **kwargs):  
        """
        Add metadata for the current move. Will be saved in the recording.
        """
        if kwargs is None:
            return
        if move_idx is None:
            move_idx = self.game.state.move_count
        self.move_metadata[move_idx] = kwargs


    def add_agent_metadata(self, **kwargs: any):
        """
        Add metadata describing the agent. Will be saved in the recording.
        """
        if kwargs:
            self.agent_metadata.update(kwargs)

    def save_metrics(self):
        """
        Save the metrics of the agent.
        """
        path = Path(self.save_metrics_path)
        path.mkdir(parents=True, exist_ok=True)

        filename = f"{self.name}_metrics.csv"
        file_path = path / filename
        data = self.metrics.to_row()
        if os.path.exists(file_path):
            with open(file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(data)
        else:
            header = self.metrics.to_header()
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerow(data)


    def save_recording(self, recording: GameRecord):
        """
        Saves a recording of a game.
        """
        recording.save(base_path=self.save_recordings_path)


    def play(self, seed: int = None, save_recording: bool = False, max_rounds: int = 0) -> GameRecord:
        """
        Play a single game.

        Args:
            save_recording: If True, the agent will save the recording of the game.
            max_rounds: Maximum number of rounds to play. If 0, the agent will play until the game is over.        
        """
        self.game = Game2048(seed=seed)
        self._after_game_init()
        if self.visualizer:
            self.visualizer.render(self.game.state)
        # We track rounds instead of using game.move_count because the move_count only counts valid moves.
        played_rounds = 0
        while not self.game.game_over and (max_rounds == 0 or played_rounds < max_rounds):
            move = self.get_move()
            move_valid, score_gained = self.game.make_move(move)
            self._after_move(move, move_valid, score_gained, self.game.state.game_over)
            played_rounds += 1

        recording = GameRecord(
            seed=self.game.seed,
            size=self.game.size,
            last_state=self.game.get_current_state(),
            agent_name=self.name,
            move_metadata=self.move_metadata,
            agent_metadata=self.agent_metadata
        )
        self.metrics.update(recording)
        if self.visualizer:
            self.visualizer.render(self.game.state)
            print(f"Game Over! Final score: {self.game.state.score}")

        if save_recording:
            self.save_recording(recording)
        
        return recording


    @abstractmethod
    def get_move(self) -> Move:
        """
        Get the next move.
        """
        pass


    def _after_move(self, move: Move, move_valid: bool, score_gained: int, game_over: bool):
        """
        Callback function called after each move was played.
        """
        if self.visualizer:
            self.visualizer.render(self.game.state)

    def _after_game_init(self):
        """
        Callback function called after a Game2048 was initialized but before the first move.
        """
        pass    
        
