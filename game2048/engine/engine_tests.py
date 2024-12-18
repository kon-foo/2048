import unittest
from array import array
from engine import Game2048, Move, GameState

class TestGame2048(unittest.TestCase):

    def setUp(self):
        self.game = Game2048(seed=42)

    def _to_array(self, data):
        """Helper to convert list to array, handles both 2D and 1D lists"""
        if not data:
            return None
        if isinstance(data[0], list):
            return [array('i', row) for row in data]
        return array('i', data)

    def _to_list(self, data):
        """Helper to convert array to list, handles both 2D and 1D arrays"""
        if not data:
            return None
        if isinstance(data[0], array):
            return [list(row) for row in data]
        return list(data)       

    def test_initial_state(self):
        state = self.game.get_current_state()
        self.assertEqual(len(state.board), 4)
        self.assertEqual(len(state.board[0]), 4)
        self.assertEqual(state.score, 0)
        self.assertFalse(state.game_over)
        non_zero_tiles = sum(tile != 0 for row in state.board for tile in row)
        self.assertEqual(non_zero_tiles, 2)

    def test_add_new_tile(self):
        state = self.game.get_current_state()
        initial_non_zero_tiles = sum(tile != 0 for row in state.board for tile in row)
        self.game._add_new_tile(state)
        new_non_zero_tiles = sum(tile != 0 for row in state.board for tile in row)
        self.assertEqual(new_non_zero_tiles, initial_non_zero_tiles + 1)

    def test_merge_line(self):
        line = [2, 2, 4, 4]
        merged_line, score = self.game._merge_line(line)
        self.assertEqual(self._to_list(merged_line), [4, 8, 0, 0])
        self.assertEqual(score, 6)

    def test_get_next_board_state(self):
        state = self.game.get_current_state()
        next_board, score_gained = self.game._get_next_board_state(state, Move.LEFT)
        self.assertEqual(len(next_board), 4)
        self.assertEqual(len(next_board[0]), 4)
        self.assertGreaterEqual(score_gained, 0)

    def test_is_valid_move(self):
        self.assertTrue(self.game.is_valid_move(Move.LEFT))
        self.assertTrue(self.game.is_valid_move(Move.RIGHT))
        self.assertTrue(self.game.is_valid_move(Move.UP))
        self.assertTrue(self.game.is_valid_move(Move.DOWN))

    def test_get_valid_moves(self):
        valid_moves = self.game.get_valid_moves()
        self.assertIn(Move.LEFT, valid_moves)
        self.assertIn(Move.RIGHT, valid_moves)
        self.assertIn(Move.UP, valid_moves)
        self.assertIn(Move.DOWN, valid_moves)

    def test_make_move(self):
        initial_state = self.game.get_current_state()
        self.assertTrue(self.game.make_move(Move.LEFT))
        new_state = self.game.get_current_state()
        self.assertNotEqual(self._to_list(initial_state.board), self._to_list(new_state.board))
        self.assertGreaterEqual(new_state.score, initial_state.score)

    def test_game_over(self):
        self.game.state.board = self._to_array([
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 4, 2]
        ])
        self.game.state.game_over = not self.game.get_valid_moves()
        self.assertTrue(self.game.state.game_over)


    def test_movement_move_validity(self):
        states_to_test = [
            {
                'board': [
                    [256, 64, 8, 2],
                    [64, 16, 32, 16],
                    [16, 2, 8, 4],
                    [8, 4, 2, 0]
                ],
                'name': "Rigt and down possible.",
                "valid_moves": [Move.RIGHT, Move.DOWN]

            },
            {
                'board': [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 2, 0, 0],
                    [0, 0, 0, 0]
                ],
                'name': "All Moves possible with empty cells",
                "valid_moves": [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
            },
            {
                'name': 'All moves possible with merging',
                'valid_moves': [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT],
                'board': [
                    [2, 2, 2, 2],
                    [2, 2, 2, 2],
                    [2, 2, 2, 2],
                    [2, 2, 2, 2]
                ]
            },
            {
                'name': 'No moves possible',
                'valid_moves': [],
                'board': [
                    [2, 4, 8, 16],
                    [16, 32, 64, 128],
                    [128, 256, 512, 1024],
                    [1024, 512, 256, 128]
                ]
            },
            {
                'name': 'all possible moves',
                'valid_moves': [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT],
                'board': [
                    [4, 32, 8, 4],
                    [16, 256, 4, 32],
                    [2, 4, 16, 8],
                    [2, 2, 8, 4]
                ]
            }
        ]
        
        for test_case in states_to_test:
            # print(f"\nTesting: {test_case['name']}")
            game = Game2048(size=4)
            game.state = GameState(
                board=self._to_array(test_case['board']),
                score=0,
                game_over=False
            )
            
            valid_moves = game.get_valid_moves()
            ## assert that each move in test_case['valid_moves'] is in valid_moves
            for move in test_case['valid_moves']:
                self.assertTrue(
                    move in valid_moves,
                    f"Move {move.value} not in valid moves"
                )
            ## assert that each move in valid_moves is in test_case['valid_moves']
            for move in valid_moves:
                self.assertTrue(
                    move in test_case['valid_moves'],
                    f"Move {move.value} should not be valid"
                )

            # Test each move
            # For each move, show what would happen
            for move in Move:
                next_board, _ = game._get_next_board_state(game.state, move)
                tiles_moved = self._did_tiles_move(self._to_list(test_case['board']), self._to_list(next_board))
                ## assert if move is in valid_moves, tiles_moved should be True
                if move in valid_moves:
                    self.assertTrue(
                        tiles_moved,
                        f"Move {move.value} should move tiles"
                    )
                ## assert if move is not in valid_moves, tiles_moved should be False
                else:
                    self.assertFalse(
                        tiles_moved,
                        f"Move {move.value} should not move tiles"
                    )
                
    
    def _did_tiles_move(self, before, after) -> bool:
        """Check if any tiles actually changed position"""
        return any(
            before[i][j] != after[i][j]
            for i in range(len(before))
            for j in range(len(before[0]))
        )      


    def test_state_changes(self):
        test_cases = [
            {
                'move': Move.LEFT,
                'board': [
                    [2, 2, 4, 4],
                    [2, 2, 4, 4],
                    [2, 2, 4, 4],
                    [2, 2, 4, 4]
                ],
                'expected_board': [
                    [4, 8, 0, 0],
                    [4, 8, 0, 0],
                    [4, 8, 0, 0],
                    [4, 8, 0, 0]
                ]
            },
            {
                'move': Move.RIGHT,
                'board': [
                    [2, 2, 4, 4],
                    [2, 2, 4, 4],
                    [2, 2, 4, 4],
                    [2, 2, 4, 4]
                ],
                'expected_board': [
                    [0, 0, 4, 8],
                    [0, 0, 4, 8],
                    [0, 0, 4, 8],
                    [0, 0, 4, 8]
                ]
            },
            {
                'move': Move.UP,
                'board': [
                    [2, 2, 2, 2],
                    [2, 2, 2, 2],
                    [4, 4, 4, 4],
                    [4, 4, 4, 4]
                ],
                'expected_board': [
                    [4, 4, 4, 4],
                    [8, 8, 8, 8],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]
                ]
            },
            {
                'move': Move.DOWN,
                'board': [
                    [2, 2, 2, 2],
                    [2, 2, 2, 2],
                    [4, 4, 4, 4],
                    [4, 4, 4, 4]
                ],
                'expected_board': [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [4, 4, 4, 4],
                    [8, 8, 8, 8]
                ]
            },
            {
                'move': Move.RIGHT,
                'board': [
                    [256, 64, 8, 2],
                    [64, 16, 32, 16],
                    [16, 2, 8, 4],
                    [8, 4, 2, 0]
                ],
                'expected_board': [
                    [256, 64, 8, 2],
                    [64, 16, 32, 16],
                    [16, 2, 8, 4],
                    [0, 8, 4, 2]
                ]
            },
            {
                'move': Move.DOWN,
                'board': [
                    [256, 64, 8, 2],
                    [64, 16, 32, 16],
                    [16, 2, 8, 4],
                    [8, 4, 2, 0]
                ],
                'expected_board': [
                    [256, 64, 8, 0],
                    [64, 16, 32, 2],
                    [16, 2, 8, 16],
                    [8, 4, 2, 4]
                ]
            },            

        ]

        for test_case in test_cases:
            next_state, _ = self.game._get_next_board_state(GameState(board=self._to_array(test_case['board'])), test_case['move'])
            self.assertEqual(self._to_list(next_state), test_case['expected_board'])


if __name__ == '__main__':
    unittest.main()