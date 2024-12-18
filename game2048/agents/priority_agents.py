from .base_agent import Agent
from game2048 import Move

class VerticalHorizontalVertical(Agent):
    """
    An agent that prefers up, over left, over right, over down.
    """

    def get_move(self) -> Move:
        valid_moves = self.game.get_valid_moves()
        if Move.UP in valid_moves:
            return Move.UP
        if Move.LEFT in valid_moves:
            return Move.LEFT
        if Move.RIGHT in valid_moves:
            return Move.RIGHT
        if Move.DOWN in valid_moves:
            return Move.DOWN
        
    def after_move(self, move, move_valid, score_gained, game_over):
        return super().after_move(move, move_valid, score_gained, game_over)

class VerticalHorizontalVerticalHorizontal(Agent):
    """
    An agent that prefers up, over left, over down, over right.
    """

    def get_move(self) -> Move:
        valid_moves = self.game.get_valid_moves()
        if Move.UP in valid_moves:
            return Move.UP
        if Move.LEFT in valid_moves:
            return Move.LEFT
        if Move.DOWN in valid_moves:
            return Move.DOWN
        if Move.RIGHT in valid_moves:
            return Move.RIGHT
    
    def after_move(self, move, move_valid, score_gained, game_over):
        return super().after_move(move, move_valid, score_gained, game_over)

class VerticalHoriontal(Agent):
    """
    An agent that prefers up, over down, over left, over right.
    """

    def _get_move(self) -> Move:
        valid_moves = self.game.get_valid_moves()
        if Move.UP in valid_moves:
            return Move.UP
        if Move.DOWN in valid_moves:
            return Move.DOWN
        if Move.LEFT in valid_moves:
            return Move.LEFT
        if Move.RIGHT in valid_moves:
            return Move.RIGHT
        

        
        