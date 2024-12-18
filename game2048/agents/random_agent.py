import random

from .base_agent import Agent
from game2048 import Move

class RandomAgent(Agent):
    """
    A random agent that makes a random move at each step.
    """

    def __init__(self, id: int = 0, use_game_rng: bool = True, **kwargs):
        """
        Initialize the agent.
        """
        super().__init__(**kwargs)
        self.id = id
        self.use_game_rng = use_game_rng
    
    @property
    def name(self) -> str:
        """
        Return the name of the agent.
        """
        return f"RandomAgent_{self.id}"

    def get_move(self) -> Move:
        """
        Return a random move.
        """
        if self.use_game_rng:
            return self.game.rng.choice(list(self.game.get_valid_moves()))
        return random.choice(list(self.game.get_valid_moves()))
    
