from abc import ABC, abstractmethod

from game2048 import GameState

class Visualizer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def render(self, state: GameState):
        pass