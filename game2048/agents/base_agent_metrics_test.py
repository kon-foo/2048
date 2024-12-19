import unittest
from game2048.agents.base_agent_metrics import AgentMetrics
from game2048 import GameRecord, GameState

class TestAgentMetrics(unittest.TestCase):

    def test_initial_metrics(self):
        metrics = AgentMetrics()
        self.assertEqual(metrics.games_played, 0)
        self.assertEqual(metrics.avg_score, 0)
        self.assertEqual(metrics.max_score, 0)
        self.assertEqual(metrics.last_score, 0)
        self.assertEqual(metrics.max_tile, 0)
        self.assertEqual(metrics.avg_max_tile, 0)
        self.assertEqual(metrics.last_max_tile, 0)
        self.assertEqual(metrics.avg_moves, 0)
        self.assertEqual(metrics.max_moves, 0)
        self.assertEqual(metrics.last_moves, 0)
        self.assertEqual(metrics.avg_invalid_moves, 0)
        self.assertEqual(metrics.max_invalid_moves, 0)
        self.assertEqual(metrics.min_invalid_moves, 0)
        self.assertEqual(metrics.last_invalid_moves, 0)
        self.assertEqual(metrics.min_invalid_move_ratio, 0)
        self.assertEqual(metrics.max_invalid_move_ratio, 0)
        self.assertEqual(metrics.total_invalid_move_ratio, 0)
        self.assertEqual(metrics.invalid_move_ratios, [])


    def test_update_metrics(self):
        metrics = AgentMetrics()
        game_record = GameRecord(
            agent_name='TestAgent',
            seed=1,
            size=4,
            last_state=GameState(
                board=[[8, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                score=16,
                ## Test that moves derived from the move_count and not the length of performed_moves to be consistent.
                performed_moves=[],
                move_count=4,
                invalid_move_count=1
                )
            )
        metrics.update(game_record)

        self.assertEqual(metrics.games_played, 1)
        self.assertEqual(metrics.avg_score, 16)
        self.assertEqual(metrics.max_score, 16)
        self.assertEqual(metrics.last_score, 16)
        self.assertEqual(metrics.max_tile, 8)
        self.assertEqual(metrics.avg_max_tile, 8)
        self.assertEqual(metrics.last_max_tile, 8)
        self.assertEqual(metrics.avg_moves, 4)
        self.assertEqual(metrics.max_moves, 4)
        self.assertEqual(metrics.last_moves, 4)
        self.assertEqual(metrics.avg_invalid_moves, 1)
        self.assertEqual(metrics.max_invalid_moves, 1)
        self.assertEqual(metrics.min_invalid_moves, 1)
        self.assertEqual(metrics.last_invalid_moves, 1)
        self.assertEqual(metrics.total_invalid_move_ratio, 0.25)
        self.assertEqual(metrics.avg_invalid_move_ratio, 0.25)
        self.assertEqual(metrics.invalid_move_ratios, [0.25])
        self.assertEqual(metrics.min_invalid_move_ratio, 0.25)
        self.assertEqual(metrics.max_invalid_move_ratio, 0.25)

    def test_multiple_updates(self):
        metrics = AgentMetrics()
        game_record1 = GameRecord(
            agent_name='TestAgent',
            seed=1,
            size=4,
            last_state=GameState(
                board=[[64, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                score=100,
                move_count=4,
                invalid_move_count=2
                )
            )
        game_record2 = GameRecord(
            agent_name='TestAgent',
            seed=1,
            size=4,
            last_state=GameState(
                board=[[32, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                score=200,
                move_count=8,
                invalid_move_count=1
                )
            )
        print(f"{metrics:short}")
        metrics.update(game_record1)
        metrics.update(game_record2)

        self.assertEqual(metrics.games_played, 2)
        ## Score Metrics
        self.assertEqual(metrics.avg_score, 150)
        self.assertEqual(metrics.max_score, 200)
        self.assertEqual(metrics.last_score, 200)

        ## Tile Metrics
        self.assertEqual(metrics.avg_max_tile, 48)
        self.assertEqual(metrics.max_tile, 64)
        self.assertEqual(metrics.last_max_tile, 32)

        ## Move Metrics
        self.assertEqual(metrics.avg_moves, 6)
        self.assertEqual(metrics.max_moves, 8)
        self.assertEqual(metrics.last_moves, 8)

        ## Invalid Move Metrics
        self.assertEqual(metrics.avg_invalid_moves, 1.5)
        self.assertEqual(metrics.max_invalid_moves, 2)
        self.assertEqual(metrics.min_invalid_moves, 1)
        self.assertEqual(metrics.last_invalid_moves, 1)
        self.assertEqual(metrics.invalid_move_ratios, [0.5, 0.125])
        self.assertEqual(metrics.min_invalid_move_ratio, 0.125)
        self.assertEqual(metrics.max_invalid_move_ratio, 0.5)
        self.assertEqual(metrics.total_invalid_move_ratio, 0.25)
        self.assertEqual(metrics.avg_invalid_move_ratio, 0.3125)

if __name__ == '__main__':
    unittest.main()