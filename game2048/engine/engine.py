import random
from typing import List, Tuple, Optional
from pydantic import BaseModel, ConfigDict, field_serializer, field_validator
from enum import IntEnum
from array import array

class Move(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class GameState(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )    
    board: List[array]
    score: int = 0
    game_over: bool = False    
    performed_moves: List[Move] = []
    move_count: int = 0

    @property
    def last_move(self) -> Optional[Move]:
        return self.performed_moves[self.move_count - 1] if self.move_count > 0 else None

    @field_serializer('board')
    def serialize_board(self, board: List[array], _info):
        return [list(row) for row in board]
    
    @field_validator('board', mode='before')
    def validate_board(cls, board):
        if not isinstance(board, list):
            raise ValueError("board must be a list of lists or array objects")

        converted_board = []
        for row in board:
            if isinstance(row, list):
                converted_board.append(array('i', row))  # 'i' for integer arrays
            elif isinstance(row, array):
                converted_board.append(row)
            else:
                raise ValueError("Each row in board must be a list or array")
        return converted_board 
    
    
class Game2048:
    def __init__(self, size: int = 4, seed: Optional[int] = None):
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        self.seed = seed
        self.rng = random.Random(seed)
        self.size = size
        self.empty_cells = set((i, j) for i in range(size) for j in range(size))
        self.state = self._create_initial_state()

    def _create_initial_state(self) -> GameState:
        """Initialize an empty board using arrays instead of lists"""
        board = [array('i', [0] * self.size) for _ in range(self.size)]
        state = GameState(board=board)
        self.empty_cells = set((i, j) for i in range(self.size) for j in range(self.size))
        
        # Add two initial tiles
        self._add_new_tile(state)
        self._add_new_tile(state)
        return state
    
    def _add_new_tile(self, state: GameState) -> bool:
        """Add a new tile using the maintained empty_cells set"""
        if not self.empty_cells:
            return False
            
        row, col = self.rng.choice(tuple(self.empty_cells))
        value = 2 if self.rng.random() < 0.9 else 4
        state.board[row][col] = value
        self.empty_cells.remove((row, col))
        return True

    def _merge_line(self, line: array) -> Tuple[array, int]:
        """Optimized merge operation working directly with arrays"""
        size = len(line)
        result = array('i', [0] * size)
        score = 0
        write_pos = 0
        i = 0
        
        # Skip leading zeros
        while i < size and line[i] == 0:
            i += 1
            
        # Process remaining numbers
        while i < size:
            current = line[i]
            if current == 0:
                i += 1
                continue
                
            # Look for next non-zero number
            next_i = i + 1
            while next_i < size and line[next_i] == 0:
                next_i += 1
                
            if next_i < size and line[next_i] == current:
                # Merge equal numbers
                merged = current * 2
                result[write_pos] = merged
                score += current
                i = next_i + 1
            else:
                # Keep single number
                result[write_pos] = current
                i = next_i
            
            write_pos += 1
            
        return result, score

    def _get_next_board_state(self, current_state: GameState, move: Move) -> Tuple[List[array], int]:
        """Apply the move to the board"""
        board = [array('i', row) for row in current_state.board]
        score_gained = 0
        
        if move in (Move.LEFT, Move.RIGHT):
            for i in range(self.size):
                line = board[i]
                if move == Move.RIGHT:
                    line.reverse()
                merged_line, score = self._merge_line(line)
                if move == Move.RIGHT:
                    merged_line.reverse()
                board[i] = merged_line
                score_gained += score
        else:  # UP or DOWN
            for j in range(self.size):
                line = array('i', (board[i][j] for i in range(self.size)))
                if move == Move.DOWN:
                    line.reverse()
                merged_line, score = self._merge_line(line)
                if move == Move.DOWN:
                    merged_line.reverse()
                for i in range(self.size):
                    board[i][j] = merged_line[i]
                score_gained += score

        return board, score_gained

    def is_valid_move(self, move: Move) -> bool:
        """Check if a move is valid in the current state"""
        next_board, score_gained = self._get_next_board_state(self.state, move)
        return score_gained > 0 or any(
            next_board[i][j] != self.state.board[i][j]
            for i in range(self.size)
            for j in range(self.size)
        )

    def make_move(self, move: Move) -> tuple[bool, int]:
        """
            Optimized move application with empty cells tracking
            Returns a tuple of (move_successful, gained_score)
        """
        ## check if move is a valid memeber of the Move Enum
        if move is None or move not in Move:
            raise ValueError("Invalid move")

        next_board, score_gained = self._get_next_board_state(self.state, move)
        
        if all(next_board[i][j] == self.state.board[i][j] 
                for i in range(self.size) 
                for j in range(self.size)):
            ## if the board is the same after the move, return False and 0
            return False, 0
            
        # Update empty cells set
        self.empty_cells = set(
            (i, j) for i in range(self.size) 
            for j in range(self.size) if next_board[i][j] == 0
        )
        
        self.state.board = next_board
        self.state.score += score_gained
        self.state.performed_moves.append(move)
        self.state.move_count += 1
        
        # Add a new tile after successful move
        tile_added = self._add_new_tile(self.state)
        
        # Check if game is over 
        if not tile_added:
            self.state.game_over = True
            return True, score_gained
            
        # Quick check for valid moves using empty cells
        if len(self.empty_cells) > 0:  # If there is an empty cell, the game is not over
            return True, score_gained
            
        # Only check for valid moves when we're close to game over
        self.state.game_over = not any(self.is_valid_move(m) for m in Move)
        return True, score_gained

    def get_valid_moves(self) -> List[Move]:
        """Return a list of all valid moves in the current state"""
        return [move for move in Move if self.is_valid_move(move)]

    def get_current_state(self) -> GameState:
        """Return a copy of the current game state"""
        return self.state.model_copy()

    def reset(self) -> None:
        """Reset the game state"""
        self.empty_cells.clear()
        self.state = self._create_initial_state()

    @property
    def game_over(self) -> bool:
        return self.state.game_over    
    
    @property
    def score(self) -> int:
        return self.state.score