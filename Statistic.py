import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from GA import GA_V6, Game, MinimaxSearchPlayer
import numpy as np
from scipy.interpolate import make_interp_spline

ga = GA_V6(60, 0.01)

win_ratio = []

population_generation = []

play_times = 6

statistic_number = 300

def play(color, params):
    game = Game(8)
    board = game.get_init_board()
    end, winner = game.check_game_end(board)
    cur_color = -1
    random_color = -color

    while not end:
        if cur_color == color:
            player = MinimaxSearchPlayer(game, board, color, 1, params)
            move = player.get_action()
            board, cur_color = game.get_next_board(board, move, color)
        else:
            legal_moves = game.get_legal_moves(board, random_color)
            move = random.choice(legal_moves)
            board, cur_color = game.get_next_board(board, move, random_color)

        end, winner = game.check_game_end(board)
    return winner


for i in tqdm(range(statistic_number), desc="reading log file"):
    file_name = str(i) + "_population.log"
    with open("log/" + file_name, 'r') as log:
        p = eval(log.read())
        population_generation.append(ga.translate_specie(p[15]))

for i in tqdm(range(statistic_number), desc="playing games"):
    win = 0
    lose = 0
    draw = 0
    for _ in range(play_times // 2):
        black = play(-1, population_generation[i])
        if black == -1:
            win += 1
        elif black == 1:
            lose += 1
        else:
            draw += 1
    for _ in range(play_times // 2):
        white = play(1, population_generation[i])
        if white == 1:
            win += 1
        elif white == -1:
            lose += 1
        else:
            draw += 1

    win_ratio.append((1.0 * win / play_times))

print(win_ratio)

x = [i for i in range(statistic_number)]
plt.xlabel("generation")
plt.ylabel("win ratio")
plt.plot(x, win_ratio)
plt.show()
