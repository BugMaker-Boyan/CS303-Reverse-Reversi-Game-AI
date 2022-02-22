import multiprocessing
import random
import time
import numpy as np
from dataclasses import dataclass

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0


class GA_V6(object):
    def __init__(self, population_size, p_mutation):
        assert population_size % 2 == 0
        self.population_size = population_size
        self.p_mutation = p_mutation
        self.population_list = []
        self.trans_population_list = []

    def generate_random_dna(self):
        dna = ""
        for i in range(314):
            dna += str(random.randint(0, 1))
        return dna

    def generate_origin_population(self):
        for i in range(self.population_size):
            self.population_list.append({"dna": self.generate_random_dna(),
                                         "generation": 0})

    def translate_gene(self, gene, negative=False):
        gene_list = list(gene)
        gene_len = len(gene_list)
        value = 0
        for i in range(gene_len):
            value += pow(2, i) * int(gene_list[i])
        if negative:
            value -= pow(2, gene_len - 1)
        return value

    def translate_specie(self, specie):
        dna = specie["dna"]
        reward_1 = []
        reward_2 = []
        cur = 0
        for i in range(10):
            reward_1.append(self.translate_gene(dna[cur:cur + 7], negative=True))
            cur += 7
        for i in range(10):
            reward_2.append(self.translate_gene(dna[cur:cur + 7], negative=True))
            cur += 7
        action_1 = []
        action_2 = []
        for i in range(10):
            action_1.append(self.translate_gene(dna[cur:cur + 7], negative=True))
            cur += 7
        for i in range(10):
            action_2.append(self.translate_gene(dna[cur:cur + 7], negative=True))
            cur += 7
        stable_weight = self.translate_gene(dna[cur:cur + 8], negative=False)
        cur += 8

        front_1 = self.translate_gene(dna[cur:cur + 6], negative=True)
        cur += 6
        front_2 = self.translate_gene(dna[cur:cur + 6], negative=True)
        cur += 6

        num_1 = self.translate_gene(dna[cur:cur + 7], negative=True)
        cur += 7
        num_2 = self.translate_gene(dna[cur:cur + 7], negative=True)
        cur += 7

        assert cur == 314

        reward_matrix_1 = np.zeros((8, 8))
        reward_matrix_1[0][0] = reward_matrix_1[0][7] = reward_matrix_1[7][0] = reward_matrix_1[7][7] = reward_1[0]
        reward_matrix_1[1][1] = reward_matrix_1[1][6] = reward_matrix_1[6][1] = reward_matrix_1[6][6] = reward_1[4]
        reward_matrix_1[2][2] = reward_matrix_1[2][5] = reward_matrix_1[5][2] = reward_matrix_1[5][5] = reward_1[7]
        reward_matrix_1[3][3] = reward_matrix_1[3][4] = reward_matrix_1[4][3] = reward_matrix_1[4][4] = reward_1[9]
        reward_matrix_1[0][1] = reward_matrix_1[1][0] = reward_matrix_1[0][6] = reward_matrix_1[1][7] = \
        reward_matrix_1[6][0] \
            = reward_matrix_1[7][1] = reward_matrix_1[6][7] = reward_matrix_1[7][6] = reward_1[1]
        reward_matrix_1[0][2] = reward_matrix_1[2][0] = reward_matrix_1[0][5] = reward_matrix_1[2][7] = \
        reward_matrix_1[5][0] \
            = reward_matrix_1[7][2] = reward_matrix_1[5][7] = reward_matrix_1[7][5] = reward_1[2]
        reward_matrix_1[0][3] = reward_matrix_1[3][0] = reward_matrix_1[0][4] = reward_matrix_1[3][7] = \
        reward_matrix_1[4][0] \
            = reward_matrix_1[7][3] = reward_matrix_1[7][4] = reward_matrix_1[4][7] = reward_1[3]
        reward_matrix_1[1][2] = reward_matrix_1[2][1] = reward_matrix_1[1][5] = reward_matrix_1[2][6] = \
        reward_matrix_1[5][1] \
            = reward_matrix_1[6][2] = reward_matrix_1[6][5] = reward_matrix_1[5][6] = reward_1[5]
        reward_matrix_1[1][3] = reward_matrix_1[3][1] = reward_matrix_1[1][4] = reward_matrix_1[3][6] = \
        reward_matrix_1[4][1] \
            = reward_matrix_1[6][3] = reward_matrix_1[6][4] = reward_matrix_1[4][6] = reward_1[6]
        reward_matrix_1[2][3] = reward_matrix_1[3][2] = reward_matrix_1[2][4] = reward_matrix_1[3][5] = \
        reward_matrix_1[4][2] \
            = reward_matrix_1[5][3] = reward_matrix_1[5][4] = reward_matrix_1[4][5] = reward_1[8]

        reward_matrix_2 = np.zeros((8, 8))
        reward_matrix_2[0][0] = reward_matrix_2[0][7] = reward_matrix_2[7][0] = reward_matrix_2[7][7] = reward_2[0]
        reward_matrix_2[1][1] = reward_matrix_2[1][6] = reward_matrix_2[6][1] = reward_matrix_2[6][6] = reward_2[4]
        reward_matrix_2[2][2] = reward_matrix_2[2][5] = reward_matrix_2[5][2] = reward_matrix_2[5][5] = reward_2[7]
        reward_matrix_2[3][3] = reward_matrix_2[3][4] = reward_matrix_2[4][3] = reward_matrix_2[4][4] = reward_2[9]
        reward_matrix_2[0][1] = reward_matrix_2[1][0] = reward_matrix_2[0][6] = reward_matrix_2[1][7] = \
            reward_matrix_2[6][0] \
            = reward_matrix_2[7][1] = reward_matrix_2[6][7] = reward_matrix_2[7][6] = reward_2[1]
        reward_matrix_2[0][2] = reward_matrix_2[2][0] = reward_matrix_2[0][5] = reward_matrix_2[2][7] = \
            reward_matrix_2[5][0] \
            = reward_matrix_2[7][2] = reward_matrix_2[5][7] = reward_matrix_2[7][5] = reward_2[2]
        reward_matrix_2[0][3] = reward_matrix_2[3][0] = reward_matrix_2[0][4] = reward_matrix_2[3][7] = \
            reward_matrix_2[4][0] \
            = reward_matrix_2[7][3] = reward_matrix_2[7][4] = reward_matrix_2[4][7] = reward_2[3]
        reward_matrix_2[1][2] = reward_matrix_2[2][1] = reward_matrix_2[1][5] = reward_matrix_2[2][6] = \
            reward_matrix_2[5][1] \
            = reward_matrix_2[6][2] = reward_matrix_2[6][5] = reward_matrix_2[5][6] = reward_2[5]
        reward_matrix_2[1][3] = reward_matrix_2[3][1] = reward_matrix_2[1][4] = reward_matrix_2[3][6] = \
            reward_matrix_2[4][1] \
            = reward_matrix_2[6][3] = reward_matrix_2[6][4] = reward_matrix_2[4][6] = reward_2[6]
        reward_matrix_2[2][3] = reward_matrix_2[3][2] = reward_matrix_2[2][4] = reward_matrix_2[3][5] = \
            reward_matrix_2[4][2] \
            = reward_matrix_2[5][3] = reward_matrix_2[5][4] = reward_matrix_2[4][5] = reward_2[8]

        action_matrix_1 = np.zeros((8, 8))
        action_matrix_1[0][0] = action_matrix_1[0][7] = action_matrix_1[7][0] = action_matrix_1[7][7] = action_1[0]
        action_matrix_1[1][1] = action_matrix_1[1][6] = action_matrix_1[6][1] = action_matrix_1[6][6] = action_1[4]
        action_matrix_1[2][2] = action_matrix_1[2][5] = action_matrix_1[5][2] = action_matrix_1[5][5] = action_1[7]
        action_matrix_1[3][3] = action_matrix_1[3][4] = action_matrix_1[4][3] = action_matrix_1[4][4] = action_1[9]
        action_matrix_1[0][1] = action_matrix_1[1][0] = action_matrix_1[0][6] = action_matrix_1[1][7] = \
            action_matrix_1[6][0] \
            = action_matrix_1[7][1] = action_matrix_1[6][7] = action_matrix_1[7][6] = action_1[1]
        action_matrix_1[0][2] = action_matrix_1[2][0] = action_matrix_1[0][5] = action_matrix_1[2][7] = \
            action_matrix_1[5][0] \
            = action_matrix_1[7][2] = action_matrix_1[5][7] = action_matrix_1[7][5] = action_1[2]
        action_matrix_1[0][3] = action_matrix_1[3][0] = action_matrix_1[0][4] = action_matrix_1[3][7] = \
            action_matrix_1[4][0] \
            = action_matrix_1[7][3] = action_matrix_1[7][4] = action_matrix_1[4][7] = action_1[3]
        action_matrix_1[1][2] = action_matrix_1[2][1] = action_matrix_1[1][5] = action_matrix_1[2][6] = \
            action_matrix_1[5][1] \
            = action_matrix_1[6][2] = action_matrix_1[6][5] = action_matrix_1[5][6] = action_1[5]
        action_matrix_1[1][3] = action_matrix_1[3][1] = action_matrix_1[1][4] = action_matrix_1[3][6] = \
            action_matrix_1[4][1] \
            = action_matrix_1[6][3] = action_matrix_1[6][4] = action_matrix_1[4][6] = action_1[6]
        action_matrix_1[2][3] = action_matrix_1[3][2] = action_matrix_1[2][4] = action_matrix_1[3][5] = \
            action_matrix_1[4][2] \
            = action_matrix_1[5][3] = action_matrix_1[5][4] = action_matrix_1[4][5] = action_1[8]

        action_matrix_2 = np.zeros((8, 8))
        action_matrix_2[0][0] = action_matrix_2[0][7] = action_matrix_2[7][0] = action_matrix_2[7][7] = action_2[0]
        action_matrix_2[1][1] = action_matrix_2[1][6] = action_matrix_2[6][1] = action_matrix_2[6][6] = action_2[4]
        action_matrix_2[2][2] = action_matrix_2[2][5] = action_matrix_2[5][2] = action_matrix_2[5][5] = action_2[7]
        action_matrix_2[3][3] = action_matrix_2[3][4] = action_matrix_2[4][3] = action_matrix_2[4][4] = action_2[9]
        action_matrix_2[0][1] = action_matrix_2[1][0] = action_matrix_2[0][6] = action_matrix_2[1][7] = \
            action_matrix_2[6][0] \
            = action_matrix_2[7][1] = action_matrix_2[6][7] = action_matrix_2[7][6] = action_2[1]
        action_matrix_2[0][2] = action_matrix_2[2][0] = action_matrix_2[0][5] = action_matrix_2[2][7] = \
            action_matrix_2[5][0] \
            = action_matrix_2[7][2] = action_matrix_2[5][7] = action_matrix_2[7][5] = action_2[2]
        action_matrix_2[0][3] = action_matrix_2[3][0] = action_matrix_2[0][4] = action_matrix_2[3][7] = \
            action_matrix_2[4][0] \
            = action_matrix_2[7][3] = action_matrix_2[7][4] = action_matrix_2[4][7] = action_2[3]
        action_matrix_2[1][2] = action_matrix_2[2][1] = action_matrix_2[1][5] = action_matrix_2[2][6] = \
            action_matrix_2[5][1] \
            = action_matrix_2[6][2] = action_matrix_2[6][5] = action_matrix_2[5][6] = action_2[5]
        action_matrix_2[1][3] = action_matrix_2[3][1] = action_matrix_2[1][4] = action_matrix_2[3][6] = \
            action_matrix_2[4][1] \
            = action_matrix_2[6][3] = action_matrix_2[6][4] = action_matrix_2[4][6] = action_2[6]
        action_matrix_2[2][3] = action_matrix_2[3][2] = action_matrix_2[2][4] = action_matrix_2[3][5] = \
            action_matrix_2[4][2] \
            = action_matrix_2[5][3] = action_matrix_2[5][4] = action_matrix_2[4][5] = action_2[8]



        return {"reward_matrix_1": reward_matrix_1,
                "reward_matrix_2": reward_matrix_2,
                "action_matrix_1": action_matrix_1,
                "action_matrix_2": action_matrix_2,
                "stable_weight": stable_weight,
                "front_1": front_1,
                "front_2": front_2,
                "num_1": num_1,
                "num_2": num_2}

    def translate_population(self):
        self.trans_population_list.clear()
        for specie in self.population_list:
            self.trans_population_list.append(self.translate_specie(specie))

    def fitness(self):
        random.shuffle(self.population_list)
        self.translate_population()
        cores = multiprocessing.cpu_count()
        with multiprocessing.Pool(processes=cores) as p:
            result = p.map(self._compete, range(self.population_size))
        p.close()
        p.join()

        for i in range(self.population_size):
            self.population_list[i]["fitness"] = result[i][0]
            self.population_list[i]["win"] = result[i][1]
            self.population_list[i]["lose"] = result[i][2]
            self.population_list[i]["draw"] = result[i][3]
        self.population_list = sorted(self.population_list, key=lambda specie: specie["fitness"], reverse=True)

    def selection(self):
        self.fitness()
        self.population_list = self.population_list[:self.population_size // 2]
        self.population_size //= 2
        assert self.population_size == len(self.population_list)
        for specie in self.population_list:
            specie["generation"] += 1

    def _compete(self, specie_index, search_depth=1):
        win = 0
        lose = 0
        draw = 0
        win_black = 0
        win_white = 0
        start_time = time.time()
        self_specie = self.trans_population_list[specie_index]
        for i in range(self.population_size):
            if i != specie_index:
                oppo_specie = self.trans_population_list[i]
                cur_color = COLOR_BLACK
                self_color = COLOR_BLACK if i % 2 == 0 else COLOR_WHITE
                oppo_color = -self_color
                game = Game(8)
                board = game.get_init_board()
                end, winner = game.check_game_end(board)
                while not end:
                    if cur_color == self_color:
                        self_ai = MinimaxSearchPlayer(game, board, self_color, search_depth, self_specie)
                        move = self_ai.get_action()
                        board, cur_color = game.get_next_board(board, move, self_color)
                    else:
                        oppo_ai = MinimaxSearchPlayer(game, board, oppo_color, search_depth, oppo_specie)
                        move = oppo_ai.get_action()
                        board, cur_color = game.get_next_board(board, move, oppo_color)
                    end, winner = game.check_game_end(board)
                if winner == self_color:
                    win += 1
                    if self_color == COLOR_BLACK:
                        win_black += 1
                    else:
                        win_white += 1
                elif winner == oppo_color:
                    lose += 1
                else:
                    draw += 1
        end_time = time.time()
        print(specie_index, end_time - start_time)
        return min(win_white, win_black), win, lose, draw

    def _cross_over(self, specie_1, specie_2):
        new_dna = ""
        dna_1 = list(specie_1["dna"])
        dna_2 = list(specie_2["dna"])
        assert len(dna_1) == len(dna_2)
        for i in range(len(dna_1)):
            if random.random() < 0.5:
                new_dna += dna_1[i]
            else:
                new_dna += dna_2[i]
        new_specie = {"dna": new_dna, "generation": 0}
        self._mutation(new_specie)
        return new_specie

    def cross_over(self):
        for i in range(self.population_size):
            specie_1 = self.population_list[i]
            random_index = np.random.choice(range(self.population_size))
            while random_index == i:
                random_index = np.random.choice(range(self.population_size))
            specie_2 = self.population_list[random_index]
            self.population_list.append(self._cross_over(specie_1, specie_2))
        self.population_size *= 2
        assert self.population_size == len(self.population_list)

    def _mutation(self, specie):
        dna = list(specie["dna"])
        for i in range(len(dna)):
            if random.random() < self.p_mutation:
                dna[i] = "1" if dna[i] == "0" else "0"
        specie["dna"] = "".join(dna)

    def train(self, total_generation):
        self.generate_origin_population()
        for i in range(total_generation):
            self.selection()
            with open(f"log/{i}_population.log", 'w') as log_file:
                log_file.write(self.population_list.__str__())
            self.cross_over()


class Game(object):
    def __init__(self, board_n):
        self.board_n = board_n
        self.directions = [(1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)]

    def get_init_board(self):
        init_board = np.zeros((self.board_n, self.board_n))
        center = self.board_n // 2
        init_board[center - 1, center - 1] = init_board[center, center] = COLOR_WHITE
        init_board[center - 1, center] = init_board[center, center - 1] = COLOR_BLACK
        return init_board

    def move_to_location(self, move):
        return move // self.board_n, move % self.board_n

    def location_to_move(self, location):
        return location[0] * self.board_n + location[1]

    def get_reverse_list(self, board, move, color):
        x, y = move
        reverse_list = []
        if board[x, y] != COLOR_NONE:
            return reverse_list

        for dx, dy in self.directions:
            dir_reverse_list = []
            dir_x, dir_y = x + dx, y + dy
            dir_reverse_flag = False
            while 0 <= dir_x < self.board_n and 0 <= dir_y < self.board_n:
                if board[dir_x, dir_y] == -color:
                    dir_reverse_list.append((dir_x, dir_y))
                    dir_x, dir_y = dir_x + dx, dir_y + dy
                elif board[dir_x, dir_y] == color:
                    dir_reverse_flag = True
                    break
                else:
                    break
            if dir_reverse_flag and len(dir_reverse_list) != 0:
                reverse_list.extend(dir_reverse_list)
        return reverse_list

    def get_next_board(self, board, move, color):
        board = np.copy(board)
        reverse_list = self.get_reverse_list(board, move, color)
        reverse_list.append(move)
        for x, y in reverse_list:
            board[x, y] = color
        if not self.has_legal_move(board, -color):
            return board, color
        else:
            return board, -color

    def get_legal_moves(self, board, color):
        legal_moves = set()
        for i in range(self.board_n):
            for j in range(self.board_n):
                if board[i, j] == color:
                    legal_moves.update(self._get_legal_move_from_location(board, (i, j), color))
        return list(legal_moves)

    def has_legal_move(self, board, color):
        for i in range(self.board_n):
            for j in range(self.board_n):
                if board[i, j] == color and self._check_legal_move_from_location(board, (i, j), color):
                    return True
        return False

    def check_game_end(self, board):
        if len(self.get_legal_moves(board, COLOR_BLACK)) == 0 and len(self.get_legal_moves(board, COLOR_WHITE)) == 0:
            board_sum = np.sum(board)
            if board_sum == 0:
                return True, 0
            elif board_sum > 0:
                return True, -1
            else:
                return True, 1
        else:
            return False, None

    def _get_legal_move_from_location(self, board, location, color):
        x, y = location
        legal_moves = set()
        for dx, dy in self.directions:
            dir_x, dir_y = x + dx, y + dy
            has_reverse_piece = False
            while 0 <= dir_x < self.board_n and 0 <= dir_y < self.board_n:
                if board[dir_x, dir_y] == -color:
                    has_reverse_piece = True
                    dir_x, dir_y = dir_x + dx, dir_y + dy
                elif board[dir_x, dir_y] == color:
                    break
                else:
                    if has_reverse_piece:
                        legal_moves.add((dir_x, dir_y))
                    break
        return legal_moves

    def _check_legal_move_from_location(self, board, location, color):
        x, y = location
        for dx, dy in self.directions:
            dir_x, dir_y = x + dx, y + dy
            has_reverse_piece = False
            while 0 <= dir_x < self.board_n and 0 <= dir_y < self.board_n:
                if board[dir_x, dir_y] == -color:
                    has_reverse_piece = True
                    dir_x, dir_y = dir_x + dx, dir_y + dy
                elif board[dir_x, dir_y] == color:
                    break
                else:
                    if has_reverse_piece:
                        return True
                    break
        return False

    def get_front_pieces_num(self, board, view_color):
        self_front_num, oppo_front_num = 0, 0
        for i in range(8):
            for j in range(8):
                if board[i][j] != COLOR_NONE:
                    for k in range(8):
                        x, y = i + self.directions[k][0], j + self.directions[k][1]
                        if 0 <= x < self.board_n and 0 <= y < self.board_n and board[x][y] == COLOR_NONE:
                            if board[i][j] == view_color:
                                self_front_num += 1
                            elif board[i][j] == -view_color:
                                oppo_front_num += 1
        return self_front_num, oppo_front_num


MAX_SEARCH = 1
MIN_SEARCH = -1
INF = float('inf')
nINF = float('-inf')


@dataclass
class Node:
    board: np.ndarray
    color: int
    search_type: int
    depth: int
    alpha: float
    beta: float
    value: float


class MinimaxSearchPlayer(object):
    def __init__(self, game, root_board, root_color, search_depth, params):
        self.game = game
        self.root_board = root_board
        self.root_color = root_color
        self.search_depth = search_depth
        self.root_node = Node(root_board, root_color, MAX_SEARCH, 0, nINF, INF, nINF)
        self.root_child_node = []
        self.root_legal_moves = []
        # return {"reward_matrix_1": reward_matrix_1,
        #         "reward_matrix_2": reward_matrix_2,
        #         "action_matrix_1": action_matrix_1,
        #         "action_matrix_2": action_matrix_2,
        #         "stable_weight": stable_weight,
        #         "front_1": front_1,
        #         "front_2": front_2,
        #         "num_1": num_1,
        #         "num_2": num_2}
        self.reward_matrix_1 = params["reward_matrix_1"]
        self.reward_matrix_2 = params["reward_matrix_2"]
        self.action_matrix_1 = params["action_matrix_1"]
        self.action_matrix_2 = params["action_matrix_2"]
        self.stable_weight = params["stable_weight"]
        self.front_1 = params["front_1"]
        self.front_2 = params["front_2"]
        self.num_1 = params["num_1"]
        self.num_2 = params["num_2"]

    def minimax_search(self, node):
        if node.depth == 1:
            self.root_child_node.append(node)
        end, winner = self.game.check_game_end(node.board)
        if end:
            if winner == self.root_color:
                node.value = INF
            else:
                node.value = nINF
            return node.value
        if node.depth == self.search_depth:
            node.value = self.get_node_value(node)
            return node.value
        next_moves = self.game.get_legal_moves(node.board, node.color)
        if node.depth == 0:
            self.root_legal_moves = next_moves
        if len(next_moves) == 0:
            next_search_type = -node.search_type
            if next_search_type == MAX_SEARCH:
                next_value = nINF
            else:
                next_value = INF
            next_node = Node(node.board, -node.color, next_search_type, node.depth + 1, node.alpha, node.beta,
                             next_value)
            back_value = self.minimax_search(next_node)
            node.value = back_value

            return node.value
        for move in next_moves:
            next_board, next_color = self.game.get_next_board(node.board, move, node.color)
            next_search_type = -node.search_type if next_color != node.color else node.search_type
            if next_search_type == MAX_SEARCH:
                next_value = nINF
            else:
                next_value = INF
            next_node = Node(next_board, next_color, next_search_type, node.depth + 1, node.alpha, node.beta,
                             next_value)
            back_value = self.minimax_search(next_node)

            if back_value == INF:
                node.value = back_value
                return node.value
            if node.search_type == MAX_SEARCH:
                node.value = max(node.value, back_value)
                if node.value >= node.beta:
                    return node.value
                node.alpha = max(node.alpha, node.value)
            else:
                node.value = min(node.value, back_value)
                if node.value <= node.alpha:
                    return node.value
                node.beta = min(node.beta, node.value)
        return node.value

    def get_node_value(self, node):
        board = node.board
        self_pieces_num = 0
        oppo_pieces_num = 0
        for i in range(8):
            for j in range(8):
                if board[i, j] == self.root_color:
                    self_pieces_num += 1
                elif board[i, j] == -self.root_color:
                    oppo_pieces_num += 1
        pieces_number = self_pieces_num + oppo_pieces_num

        game_state = int((pieces_number - 5) / 30)
        stable_matrix = np.zeros((8, 8))
        board_reward_score = 0
        action_score = 0
        stable_score = 0
        front_score = 0
        pieces_num_score = 0

        if board[0, 0] != COLOR_NONE:
            check_color = board[0, 0]
            width = 0
            for i in range(8):
                if board[0, i] == check_color:
                    width = i
                    stable_matrix[0][i] = 1
                else:
                    break
            for i in range(8):
                if board[i, 0] != check_color:
                    break
                for j in range(width):
                    if board[i, j] == check_color:
                        stable_matrix[i][j] = 1
                        width = j
                    else:
                        break
                if width == 0:
                    width = 1
            depth = 0
            for i in range(8):
                if board[i, 0] == check_color:
                    depth = i
                    stable_matrix[i][0] = 1
                else:
                    break
            for j in range(8):
                if board[0, j] != check_color:
                    break
                for i in range(depth):
                    if board[i, j] == check_color:
                        stable_matrix[i, j] = 1
                        depth = i
                    else:
                        break
                if depth == 0:
                    depth = 1

        if board[0, 7] != COLOR_NONE:
            check_color = board[0, 7]

            width = 7
            for j in range(7, -1, -1):
                if board[0, j] == check_color:
                    stable_matrix[0, j] = 1
                    width = j
                else:
                    break
            for i in range(8):
                if board[i, 7] != check_color:
                    break
                for j in range(7, width, -1):
                    if board[i, j] == check_color:
                        stable_matrix[i, j] = 1
                        width = j
                    else:
                        break
                if width == 7:
                    width -= 1
            depth = 0
            for i in range(8):
                if board[i, 7] == check_color:
                    stable_matrix[i, 7] = 1
                    depth = i
                else:
                    break
            for j in range(7, -1, -1):
                if board[0, j] != check_color:
                    break
                for i in range(depth):
                    if board[i, j] == check_color:
                        stable_matrix[i, j] = 1
                        depth = i
                    else:
                        break
                if depth == 0:
                    depth += 1

        if board[7, 0] != COLOR_NONE:
            check_color = board[7, 0]

            width = 0
            for j in range(8):
                if board[7, j] == check_color:
                    width = j
                    stable_matrix[7, j] = 1
                else:
                    break
            for i in range(7, -1, -1):
                if board[i, 0] != check_color:
                    break
                for j in range(width):
                    if board[i, j] == check_color:
                        stable_matrix[i, j] = 1
                        width = j
                    else:
                        break
                if width == 0:
                    width += 1
            depth = 0
            for i in range(7, -1, -1):
                if board[i, 0] == check_color:
                    depth = i
                    stable_matrix[i, 0] = 1
                else:
                    break
            for j in range(8):
                if board[7, j] != check_color:
                    break
                for i in range(7, depth, -1):
                    if board[i, j] == check_color:
                        stable_matrix[i, j] = 1
                        depth = i
                if depth == 7:
                    depth -= 1

        if board[7, 7] != COLOR_NONE:
            check_color = board[7, 7]

            width = 7
            for j in range(7, -1, -1):
                if board[7, j] == check_color:
                    width = j
                    stable_matrix[7, j] = 1
                else:
                    break
            for i in range(7, -1, -1):
                if board[i, 7] != check_color:
                    break
                for j in range(7, width, -1):
                    if board[i, j] == check_color:
                        stable_matrix[i, j] = 1
                        width = j
                    else:
                        break
                if width == 7:
                    width -= 1
            depth = 7
            for i in range(7, -1, -1):
                if board[i, 7] == check_color:
                    depth = i
                    stable_matrix[i, 7] = 1
                else:
                    break
            for j in range(7, -1, -1):
                if board[7, j] != check_color:
                    break
                for i in range(7, depth, -1):
                    if board[i, j] == check_color:
                        stable_matrix[i, j] = 1
                        depth = i
                    else:
                        break
                if depth == 7:
                    depth -= 1

        self_stable = 0
        oppo_stable = 0

        for i in range(8):
            for j in range(8):
                if board[i, j] == self.root_color:
                    if game_state == 0:
                        board_reward_score -= self.reward_matrix_1[i][j]
                    else:
                        board_reward_score -= self.reward_matrix_2[i][j]
                    if stable_matrix[i, j] == 1:
                        self_stable += 1
                elif board[i, j] == -self.root_color:
                    if game_state == 0:
                        board_reward_score += self.reward_matrix_1[i][j]
                    else:
                        board_reward_score += self.reward_matrix_2[i][j]
                    if stable_matrix[i, j] == 1:
                        oppo_stable += 1
        stable_score = (oppo_stable - self_stable) * self.stable_weight

        self_actions = self.game.get_legal_moves(board, self.root_color)
        oppo_actions = self.game.get_legal_moves(board, -self.root_color)
        for x, y in self_actions:
            if game_state == 0:
                action_score += self.action_matrix_1[x][y]
            else:
                action_score += self.action_matrix_2[x][y]
        for x, y in oppo_actions:
            if game_state == 0:
                action_score -= self.action_matrix_1[x][y]
            else:
                action_score -= self.action_matrix_2[x][y]

        self_front, oppo_front = self.game.get_front_pieces_num(board, self.root_color)
        if game_state == 0:
            front_score = (self_front - oppo_front) * self.front_1
            pieces_num_score = (self_pieces_num - oppo_pieces_num) * self.num_1
        else:
            front_score = (self_front - oppo_front) * self.front_2
            pieces_num_score = (self_pieces_num - oppo_pieces_num) * self.num_2

        return board_reward_score + action_score + stable_score + front_score + pieces_num_score

    def get_action(self):
        max_val = nINF
        best_move = None
        self.minimax_search(self.root_node)
        for i, node in enumerate(self.root_child_node):
            if max_val <= node.value:
                max_val = node.value
                best_move = self.root_legal_moves[i]
            if max_val == INF:
                break
        return best_move


class FinalSearch(object):
    def __init__(self, game, root_color):
        self.game = game
        self.root_color = root_color
        self.best_move = None

    def search(self, board, color, depth):
        if color == self.root_color:
            legal_moves = self.game.get_legal_moves(board, color)
            move_flag_list = []
            for move in legal_moves:
                next_board, next_color = self.game.get_next_board(board, move, color)
                end, winner = self.game.check_game_end(next_board)
                if end:
                    if winner == self.root_color:
                        if depth == 0:
                            self.best_move = move
                        return 1
                    elif winner == -self.root_color:
                        move_flag_list.append(-1)
                        continue
                    else:
                        move_flag_list.append(0)
                        continue
                flag = self.search(next_board, next_color, depth + 1)
                move_flag_list.append(flag)
                if flag == 1:
                    if depth == 0:
                        self.best_move = move
                    return 1
            for i in range(len(move_flag_list)):
                if move_flag_list[i] == 0:
                    if depth == 0:
                        self.best_move = legal_moves[i]
                    return 0
            return -1
        else:
            legal_moves = self.game.get_legal_moves(board, color)
            move_flag_list = []
            for move in legal_moves:
                next_board, next_color = self.game.get_next_board(board, move, color)
                end, winner = self.game.check_game_end(next_board)
                if end:
                    if winner == -self.root_color:
                        return -1
                    elif winner == self.root_color:
                        move_flag_list.append(1)
                        continue
                    else:
                        move_flag_list.append(0)
                        continue
                flag = self.search(next_board, next_color, depth + 1)
                move_flag_list.append(flag)
                if flag == -1:
                    return -1
            for flag in move_flag_list:
                if flag == 0:
                    return 0
            return 1


if __name__ == '__main__':
    ga = GA_V6(population_size=60,
               p_mutation=0.05)
    # ga.train(100000)

    print(ga.translate_specie(
        {'dna': '01111111000010000110110010100001011111001110111110110011100001000001000110011100010011011010011011000010110001111000011010000100100110010100111110111010110011100000111010111011001000011011110000110111100011010010011000101111111011101100110110111010101100101010001000000010011000011101011100000010110000010000101100', 'generation': 5, 'fitness': 18, 'win': 39, 'lose': 18, 'draw': 2}))
