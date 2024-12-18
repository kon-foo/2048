import random
import time
from game2048.engine import Game2048, Move

def get_random_move():
    return random.choice(list(Move))

def test_performance(moves=10000):
    move_count = 0
    game_count = 0
    game = Game2048(size=4, seed=0)
    for i in range(moves):
        if game.state.game_over:
            game_count += 1
            game = Game2048(size=4, seed=0)
        move = get_random_move()
        game.make_move(move)
        move_count += 1
    return move_count, game_count

if __name__ == "__main__":
    MOVES_TO_TEST = 100000
    start = time.time()
    moves, games = test_performance(moves=MOVES_TO_TEST)
    time_elapsed = time.time() - start
    # At least 10,000/second on average
    assert time_elapsed < MOVES_TO_TEST/10000, f"Performance test failed: {time_elapsed:.2f} seconds"
    print(f"Performance test passed: {time_elapsed:.2f} seconds in {games} games and {moves} moves")