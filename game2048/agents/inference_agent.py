from typing import Optional, Dict
import torch
import torch.nn as nn
import numpy as np
from array import array
from typing import List
from pathlib import Path
import time

from .base_agent import Agent
from game2048 import Move

class InferenceAgent(Agent):
    """
    Agent that uses a pre-trained model to play 2048.
    Records confidence scores for each move as metadata.
    """
    def __init__(
        self,
        model_path: str,
        network_class: nn.Module,
        moves_per_second: int = 3,
        self_unstuck: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Load the saved model
        self.policy_net = network_class()
        checkpoint = torch.load(model_path, weights_only=True)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.policy_net.eval()  # Set to evaluation mode
        self.seconds_per_move = 1 / moves_per_second if moves_per_second != 0 else 0
        self.manual = True if moves_per_second == 0 else False
        self.self_unstuck = self_unstuck
        self.previous_move: Move = None
        self.previous_move_valid = True

    def game_board_to_tensor(self, board: List[array]) -> torch.Tensor:
        """
            Convert board to neural network input using numpy intermediary. Applies a log2 transformation to normalize the values.
        """
        # Convert to numpy array in one go
        state = np.frombuffer(b''.join(row.tobytes() for row in board), dtype=np.int32).reshape(4, 4)
        # Apply log2 where values > 0
        mask = state > 0
        state = state.astype(np.float32)  # Convert once to float32
        state[mask] = np.log2(state[mask])
        return torch.from_numpy(state).flatten()           
        

    def get_move(self) -> Optional[Move]:
        """Get the next move using the trained model"""
        
        with torch.no_grad():
            # Get model predictions
            move_scores = self.policy_net(self.game_board_to_tensor(self.game.state.board))
            # Filter out invalid moves
            if self.self_unstuck and not self.previous_move_valid:
                move_scores[self.previous_move] = float('-inf')

            move_idx = move_scores.argmax().item()
            
            return Move(move_idx)
    
    def _after_move(self, move, move_valid, score_gained, game_over):
        """
        Record the confidence scores for the chosen move.
        """
        self.previous_move = move
        self.previous_move_valid = move_valid
        if self.manual:
            input("Press Enter to continue...")
        else:
            time.sleep(self.seconds_per_move)
        return super()._after_move(move, move_valid, score_gained, game_over)

