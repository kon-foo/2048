import json
import time
from pathlib import Path
from typing import Any, Dict
from pydantic import BaseModel

from game2048.engine.engine import GameState

class GameRecord(BaseModel):
    timestamp: int = int(time.time())
    seed: int
    size: int
    agent_name: str
    last_state: GameState
    move_metadata: Dict[int, Dict[str, Any]] = {}
    agent_metadata: Dict[str, Any] = {}

    @property
    def score(self) -> int:
        return self.last_state.score
    
    @property
    def max_tile(self) -> int:
        return max(max(row) for row in self.last_state.board)
    
    @property
    def move_count(self) -> int:
        return self.last_state.move_count
    

    def save(self, base_path: str = "recordings", filename: str = None):
        """
        Save the record to a file.
        """
        if filename is None:
            filename = f"{self.agent_name}_{self.score}_{self.seed}.json"
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)
        with open(f"{base_path}/{filename}", "w") as f:
            json.dump(self.model_dump(), f)
