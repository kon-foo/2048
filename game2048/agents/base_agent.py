from abc import ABC, abstractmethod

from game2048 import Game2048, Move, GameRecord
from game2048.visualize import Visualizer


class AgentMetrics:
    games_played = 0
    total_moves = []
    invalid_moves = []
    max_tiles = []
    scores = []

    ## Scores
    @property
    def avg_score(self):
        return sum(self.scores) / self.games_played if self.scores and self.games_played > 0 else 0
    
    @property
    def max_score(self):
        return max(self.scores) if self.scores else 0
    
    @property
    def last_score(self):
        return self.scores[-1] if self.scores else 0
    
    ## Max Tiles
    @property
    def max_tile(self):
        return max(self.max_tiles) if self.max_tiles else 0
    
    @property
    def avg_max_tile(self):
        return sum(self.max_tiles) / self.games_played if self.max_tiles and self.games_played > 0 else 0
    
    @property
    def last_max_tile(self):
        return self.max_tiles[-1] if self.max_tiles else 0
    
    ## Move Counts
    @property
    def avg_moves(self):
        return sum(self.total_moves) / self.games_played if self.total_moves and self.games_played > 0 else 0
    
    @property
    def max_moves(self):
        return max(self.total_moves) if self.total_moves else 0
    
    @property
    def last_moves(self):
        return self.total_moves[-1] if self.total_moves else 0
    
    ## Invalid moves
    @property
    def avg_invalid_moves(self):
        return sum(self.invalid_moves) / self.games_played if self.invalid_moves and self.games_played > 0 else 0
    
    @property
    def max_invalid_moves(self):
        return max(self.invalid_moves) if self.invalid_moves else 0
    
    @property
    def min_invalid_moves(self):
        return min(self.invalid_moves) if self.invalid_moves else 0
    
    @property
    def last_invalid_moves(self):
        return self.invalid_moves[-1] if self.invalid_moves else 0
    
    @property
    def invalid_move_ratio(self):
        return self.avg_invalid_moves / self.avg_moves if self.avg_moves > 0 else 0
    

    def update(self, game_record: GameRecord):
        self.games_played += 1
        self.total_moves.append(game_record.move_count)
        self.invalid_moves.append(game_record.invalid_move_count)
        self.max_tiles.append(game_record.max_tile)
        self.scores.append(game_record.last_state.score)

    




class Agent(ABC):
    """
    An Agent can have a lifetime of multiple games.
    An Agent runs a single game at a time.
    It calls its own _get_move method to get the next move and then calls makes the move in the game.
    Base interface for all agents.
    """

    def __init__(self, save_base_path: str = "recordings", visualizer: Visualizer = None):
        """
        Initialize the agent.
        """

        self.game: Game2048 = None
        self.save_base_path = save_base_path
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
    

    def _save_recording(self):
        """
        Save the recording of the game.
        """
        self.game.save_recording(self.name)


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
            recording.save(base_path=self.save_base_path)

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
        
