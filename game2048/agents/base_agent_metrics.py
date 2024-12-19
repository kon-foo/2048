from dataclasses import dataclass, field, fields
from game2048 import GameRecord, Move
import inspect


@dataclass
class AgentMetrics:
    games_played = 0
    total_moves: list[int] = field(default_factory=list)
    invalid_moves: list[int] = field(default_factory=list)
    max_tiles: list[int] = field(default_factory=list)
    scores: list[int] = field(default_factory=list)

    # Fields and properties to ignore in to_row and to_header
    _default_ignored_fields_and_properties = {'invalid_move_ratios', 'total_moves', 'invalid_moves', 'max_tiles', 'scores'}

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
    def avg_max_tile(self):
        return sum(self.max_tiles) / self.games_played if self.max_tiles and self.games_played > 0 else 0
    
    @property
    def max_tile(self):
        return max(self.max_tiles) if self.max_tiles else 0
    
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
    def total_invalid_move_ratio(self):
        return sum(self.invalid_moves) / sum(self.total_moves) if self.total_moves else 0
    
    @property
    def avg_invalid_move_ratio(self):
        return sum(self.invalid_move_ratios) / len(self.invalid_move_ratios) if self.invalid_move_ratios else 0
    
    @property
    def invalid_move_ratios(self):
        return [invalid / total for invalid, total in zip(self.invalid_moves, self.total_moves)] if self.total_moves else [] 
    
    @property
    def min_invalid_move_ratio(self):
        return min(self.invalid_move_ratios) if self.invalid_move_ratios else 0
    
    @property
    def max_invalid_move_ratio(self):
        return max(self.invalid_move_ratios) if self.invalid_move_ratios else 0

    
    def update(self, game_record: GameRecord):
        self.games_played += 1
        self.total_moves.append(game_record.last_state.move_count)
        self.invalid_moves.append(game_record.last_state.invalid_move_count)
        self.scores.append(game_record.last_state.score)
        self.max_tiles.append(game_record.max_tile)


    def _non_ignored_fields_and_properties(self, ignore: set[str] = set(), include: set[str] = set()) -> set[str]:
        """Returns a set of all property and field names except for the properties/fields in self._default_ignored_fields_and_properties.
            Args:
                ignore: List of further property/field names to ignore.
                include: List of property/field names to include to override ignore.
        """
        # Combine default ignored and user-specified ignore list
        ignore_set = self._default_ignored_fields_and_properties.union(ignore) - include

        # Collect field names
        field_names = {f.name for f in fields(self)}

        # Collect property names
        property_names = {
            name
            for name, member in inspect.getmembers(self.__class__)
            if isinstance(member, property)
        }

        # Filter out ignored items
        all_names = field_names.union(property_names)
        filtered_names = {name for name in all_names if name not in ignore_set}

        ## sort to ensure consistent order
        return sorted(filtered_names)

    def to_row(self, ignore: set[str] = set(), include: set[str] = set()) -> list:
        """Outputs all property values and field values in an array. Except for the properties/fields in self._default_ignored_fields_and_properties.
            Args:
                ignore: List of further property/field names to ignore
                include: List of property/field names to include to override ignore.
        """
        filtered_values = {name: getattr(self, name) for name in self._non_ignored_fields_and_properties(ignore, include)}
        return list(filtered_values.values())        
    

    def to_header(self, ignore: set[str] = set(), include: set[str] = set()) -> list[str]:
        """Outputs all property and field names as strings in an array. Except for the properties/fields in self._default_ignored_fields_and_properties."
            Args:
                ignore: List of further property/field names to ignore.
                include: List of property/field names to include to override ignore.
        """
        return list(self._non_ignored_fields_and_properties(ignore, include))
 

    
    def __str__(self):
        header = self.to_header()
        row = self.to_row()
        return "\n".join([f"{header[i]}: {row[i]}" for i in range(len(header))])
    
    def __format__(self, format_spec):
        if format_spec == "short":
            return f"Games Played: {self.games_played}\n--------------------\nAvg Score:    {self.avg_score:.2f}\nAvg Moves:    {self.avg_moves:.2f}\nTotal IMR:    {self.total_invalid_move_ratio:.2f}\nMin IMR:      {self.min_invalid_move_ratio:.2f}\n"
        else:
            return self.__str__()