from pydantic import BaseModel
import torch
import random
from array import array
from typing import List, Optional
import numpy as np
from collections import deque

from ..base_agent import Agent
from game2048 import Move, GameState
from .dqn import PrioritizedExperienceBuffer, ExperienceBuffer, BiasedExperienceBuffer

from game2048.visualize import ConsoleVisualizer

class DQNAgentConfig(BaseModel):
    """
    Configuration for a DQNAgent.
    """
    epsilon_initial: float = 1.0
    epsilon_final: float = 0.1
    # TODO: Switch to moves to play for more consistency
    games_to_play: int = 10
    invalid_move_penalty: float = -1.0

    training_frequency: int = 64 # After how many moves to train
    target_update_frequency_as_a_multiple_of_training: int = 100
    memory_capacity: int = 10000
    memory_batch_size: int = 64
    optim_lr: float = 0.001
    gamma: float = 0.99


class DQNAgent(Agent):
    """
    A DQN agent that trains using deep Q-learning.
    """
    
    def __init__(self, 
                 config: DQNAgentConfig, 
                 network_class: torch.nn.Module,
                 generation: int = 0, id: str = "0"):
        """
        Initialize the agent.
        """
        super().__init__()
        self.id = id
        self.generation = generation
        self.config = config

        ## Training specific attributes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_net = network_class().to(self.device)
        self.policy_net = network_class().to(self.device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=config.optim_lr)
        self.memory = ExperienceBuffer(capacity=config.memory_capacity)
        # self.memory = BiasedExperienceBuffer(alpha=0.6, reward_offset=config.invalid_move_penalty*-1, capacity=config.memory_capacity)
        # self.memory = PrioritizedExperienceBuffer(capacity=config.memory_capacity, invalid_move_ratio=0.2)

        ## Variables
        self.games_played = 0
        self.move_across_games = 0   
        self.current_board_as_tensor = None
        # Do-not-make-the-same-mistake-twice policy
        self.previous_move_valid: bool = True
        self.previous_move: Move = None
        self.epsilon = config.epsilon_initial 
        self.losses = deque(maxlen=1000)

    
    @property
    def name(self) -> str:
        """
        Return the name of the agent.
        """
        return f"DQNAgent_Gen{self.generation}_ID{self.id}"
    
    ## Learning specific methods
    def initialize_networks(self, state_dict: dict[str, torch.Tensor]):
        """
        Initialize the policy and target networks and the optimizer.
        """
        self.policy_net.load_state_dict(state_dict['policy_net'])
        self.target_net.load_state_dict(state_dict['target_net'])
        self.optimizer.load_state_dict(state_dict['optimizer'])

    def _game_board_to_tensor(self, board: List[array]) -> torch.Tensor:
        """
            Convert board to neural network input using numpy intermediary. Applies a log2 transformation to normalize the values.
        """
        # Convert to numpy array in one go
        state = np.frombuffer(b''.join(row.tobytes() for row in board), dtype=np.int32).reshape(4, 4)
        # Apply log2 where values > 0
        mask = state > 0
        state = state.astype(np.float32)  # Convert once to float32
        state[mask] = np.log2(state[mask])
        return torch.from_numpy(state).flatten().to(self.device)

    def _calculate_reward(self, valid_move: bool, score_gained: int) -> float:  
        """
        Calculate the reward for the move.
        Returns config.invalid_move_penalty for invalid moves, otherwise the log2 of the score gained.
        """
        if not valid_move:
            return self.config.invalid_move_penalty
        # Scale down the rewards to be more comparable
        # return 0.1 * np.log2(score_gained) if score_gained > 0 else 0
        return np.log2(score_gained) if score_gained > 0 else 0

    def train(self):
        """
        Perform Q-learning.
        """
        if len(self.memory) < self.config.memory_batch_size:
            return  
        # Sample from memory
        batch = self.memory.sample(self.config.memory_batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert tuples to batched tensors
        states = torch.stack(states).to(self.device)
        actions = torch.stack(actions).to(self.device)
        rewards = torch.stack(rewards).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.stack(dones).to(self.device)        

        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute next Q values using target net for stability
        with torch.no_grad():
            ## Single Q-value
            max_next_q_values = self.target_net(next_states).max(1)[0]
            expected_q_values = rewards + (self.config.gamma * max_next_q_values * (1 - dones))    

            # ## Double Q-value
            # next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            # next_q_values = self.target_net(next_states).gather(1, next_actions)
            # expected_q_values = rewards + (self.config.gamma * next_q_values.squeeze()   
             
            ## Q-Value Normalization
            expected_q_values = expected_q_values - expected_q_values.mean()
            expected_q_values = expected_q_values / (expected_q_values.std() + 1e-8)                     

            if self.move_across_games % (self.config.training_frequency * self.config.target_update_frequency_as_a_multiple_of_training) == 0:
                print(f"Q-values min: {current_q_values.min().item():.2f}, "
                    f"max: {current_q_values.max().item():.2f}, "
                    f"mean: {current_q_values.mean().item():.2f}")
                print(f"Expected Q-values min: {expected_q_values.min().item():.2f}, "
                    f"max: {expected_q_values.max().item():.2f}, "
                    f"mean: {expected_q_values.mean().item():.2f}")    
                   
        
        loss = torch.nn.MSELoss()(current_q_values.squeeze(), expected_q_values)
        # Clip gradients

        # Clear previous gradients
        self.optimizer.zero_grad()
        # Compute gradients
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)    
        # torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1.0)  
        # loss = torch.clamp(loss, -100, 100)

        # Update weights 
        self.optimizer.step()

        return loss.item()        

    def save(self, filename: Optional[str] = None):
        """
        Save NN weights and optimizer state to a file.
        """
        if filename is None:
            filename = f"{self.name}.pt"
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config.model_dump()}
            , f"{self.save_base_path}/{filename}")

    

    def get_move(self) -> Move:
        """
        Return a move based on epsilon-greedy policy, but avoid repeating the an invalid move.
        """
        if random.random() < self.epsilon:
            # Random move but avoid the last invalid move if there was one
            available_moves = [move for move in Move if self.previous_move_valid or move != self.previous_move]
            return random.choice(available_moves)
        else:
            with torch.no_grad():
                q_values = self.policy_net(
                    self.current_board_as_tensor
                )
                
                # If we have a last invalid move, temporarily discourage it
                if not self.previous_move_valid:
                    q_values[self.previous_move.value] = -np.inf
                    
                return Move(q_values.argmax().item())     
    
    def _after_move(self, move: Move, move_valid: bool, score_gained: int, game_over: bool):
        """
        After a move, calculate the reward, store the transition in memory and train the network.
        """
        self.move_across_games += 1
        reward = self._calculate_reward(move_valid, score_gained)
        board_after_move_as_tensor = self._game_board_to_tensor(self.game.state.board)
        if self.current_board_as_tensor is not None:
            self.memory.add(self.current_board_as_tensor,  
                            torch.tensor(move.value).to(self.device),
                            torch.tensor(reward, dtype=torch.float32).to(self.device), 
                            board_after_move_as_tensor,
                            torch.tensor(game_over, dtype=torch.float32).to(self.device))
        self.current_board_as_tensor = board_after_move_as_tensor
        if self.move_across_games % self.config.training_frequency == 0:
            loss = self.train()
            if loss is not None:
                self.losses.append(loss)
        if self.move_across_games % (self.config.training_frequency * self.config.target_update_frequency_as_a_multiple_of_training) == 0:
            ## Hard update
            # self.target_net.load_state_dict(self.policy_net.state_dict())
            ## Soft update
            tau = 0.001
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(tau * policy_param.data + (1 - tau) * target_param.data)            
            avg_loss = sum(self.losses) / len(self.losses)
            print(f"Average loss: {avg_loss:.2f}")  
            print(f"Games played: {self.games_played}")
            print(f"")

    def _after_game_init(self):
        """
        Reset the current board and previous move.
        """
        self.current_board_as_tensor = self._game_board_to_tensor(self.game.state.board)
        self.previous_move = None
        self.previous_move_valid = True

    def play_many(self):
        """
        Play config.games_to_play games. Reduce epsilon after each game.
        """
        best_score = 0
        for round in range(self.config.games_to_play):
            record = self.play(max_rounds=1000)
            self.games_played += 1
            if record.score > best_score:
                best_score = record.score
                record.save(base_path=self.save_base_path)
                # Save current model
                self.save(filename=f"{self.name}_best.pt")
                print(f"New best score: {best_score}")
            self.epsilon = max(self.config.epsilon_final, self.epsilon - (self.config.epsilon_initial - self.config.epsilon_final) / self.config.games_to_play)

