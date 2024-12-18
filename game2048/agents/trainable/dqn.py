from torch import nn
import random
import numpy as np
from collections import deque
from typing import List, Tuple
from torch import Tensor


 

class ExperienceBuffer:
    """Stores experiences that we can learn from"""
    def __init__(self, capacity: int = 10000):
        self.memory = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """Store a transition"""
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """Random sample of experiences"""
        return random.sample(self.memory, min(batch_size, len(self.memory)))
    
    def __len__(self):
        return len(self.memory)    
    

class PrioritizedExperienceBuffer:
    """Experience buffer with balanced sampling and optional global sharing"""
    def __init__(self, capacity: int = 10000, invalid_move_ratio: float = 0.2):
        self.valid_moves = deque(maxlen=int(capacity * (1 - invalid_move_ratio)))
        self.invalid_moves = deque(maxlen=int(capacity * invalid_move_ratio))
        self.invalid_move_ratio = invalid_move_ratio

    def add(self, state: Tensor, action: int, reward: int, next_state: Tensor, done: bool):
        """Store a transition"""
        is_invalid = reward < 0
        experience = (state, action, reward, next_state, done)
        if is_invalid:
            self.invalid_moves.append(experience)
        else:
            self.valid_moves.append(experience)
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """Sample with balanced ratio of valid/invalid moves"""
        # Determine number of invalid moves to sample
        n_invalid = min(
            int(batch_size * self.invalid_move_ratio),
            len(self.invalid_moves)
        )
        n_valid = min(
            batch_size - n_invalid,
            len(self.valid_moves)
        )
        
        # Adjust invalid samples if we don't have enough valid moves
        if n_valid < (batch_size - n_invalid):
            n_invalid = min(batch_size - n_valid, len(self.invalid_moves))
        
        # Sample from both buffers
        invalid_sample = random.sample(list(self.invalid_moves), n_invalid) if n_invalid > 0 else []
        valid_sample = random.sample(list(self.valid_moves), n_valid) if n_valid > 0 else []
        
        # Combine and shuffle
        combined = invalid_sample + valid_sample
        random.shuffle(combined)
        return combined
    
    @property
    def memory(self):
        return list(self.valid_moves) + list(self.invalid_moves)
    
    def __len__(self):
        return len(self.valid_moves) + len(self.invalid_moves)    
    

class BiasedExperienceBuffer:
    """Stores experiences with prioritized sampling based on rewards"""
    def __init__(self, capacity: int = 10000, alpha: float = 0.6, reward_offset: float = 1.0):
        """
        Initialize buffer with priority parameters
        
        Args:
            capacity: Maximum number of experiences to store
            alpha: Priority exponent (0 = uniform sampling, 1 = fully prioritized)
            reward_offset: Added to reward to ensure positive priorities (for negative rewards)
        """
        self.memory = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha
        self.reward_offset = reward_offset
        
    def add(self, state, action, reward, next_state, done):
        """Store a transition with its priority"""
        self.memory.append((state, action, reward, next_state, done))
        
        # Calculate priority based on absolute reward value
        priority = (abs(reward) + self.reward_offset) ** self.alpha
        self.priorities.append(priority)
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """Sample experiences based on their priorities"""
        batch_size = min(batch_size, len(self.memory))
        
        if batch_size == 0:
            return []
            
        # Convert priorities to numpy array for efficient sampling
        priorities = np.array(self.priorities)
        
        # Calculate sampling probabilities
        probs = priorities / priorities.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(
            len(self.memory),
            batch_size,
            p=probs,
            replace=False
        )
        
        # Return sampled experiences
        return [self.memory[idx] for idx in indices]
    
    def __len__(self):
        return len(self.memory)
    
    def get_priority_stats(self):
        """Return statistics about priorities for monitoring"""
        if not self.priorities:
            return {"min": 0, "max": 0, "mean": 0}
        priorities = np.array(self.priorities)
        return {
            "min": priorities.min(),
            "max": priorities.max(),
            "mean": priorities.mean()
        }    