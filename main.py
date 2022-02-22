from dataclasses import dataclass
import numpy as np
import time
import random
import timeout_decorator


COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
random.seed()
params = {"reward_matrix_1": [[47, -35, 9, 23, 23, 9, -35, 47], [-35, -25, 48, 25, 25, 48, -25, -35], [9, 48, 6, -45, -45, 6, 48, 9], [23, 25, -45, -53, -53, -45, 25, 23], [23, 25, -45, -53, -53, -45, 25, 23], [9, 48, 6, -45, -45, 6, 48, 9], [-35, -25, 48, 25, 25, 48, -25, -35], [47, -35, 9, 23, 23, 9, -35, 47]],
                "reward_matrix_2": [[-18, -22, 39, -5, -5, 39, -22, -18], [-22, 62, 16, -41, -41, 16, 62, -22], [39, 16, -10, -35, -35, -10, 16, 39], [-5, -41, -35, 16, 16, -35, -41, -5], [-5, -41, -35, 16, 16, -35, -41, -5], [39, 16, -10, -35, -35, -10, 16, 39], [-22, 62, 16, -41, -41, 16, 62, -22], [-18, -22, 39, -5, -5, 39, -22, -18]],
                "action_matrix_1": [[55, 56, 39, 25, 25, 39, 56, 55], [56, 27, -11, 60, 60, -11, 27, 56], [39, -11, 52, 3, 3, 52, -11, 39], [25, 60, 3, -41, -41, 3, 60, 25], [25, 60, 3, -41, -41, 3, 60, 25], [39, -11, 52, 3, 3, 52, -11, 39], [56, 27, -11, 60, 60, -11, 27, 56], [55, 56, 39, 25, 25, 39, 56, 55]],
                "action_matrix_2": [[-22, 12, 43, 56, 56, 43, 12, -22], [12, -51, 49, 41, 41, 49, -51, 12], [43, 49, 55, 36, 36, 55, 49, 43], [56, 41, 36, -60, -60, 36, 41, 56], [56, 41, 36, -60, -60, 36, 41, 56], [43, 49, 55, 36, 36, 55, 49, 43], [12, -51, 49, 41, 41, 49, -51, 12], [-22, 12, 43, 56, 56, 43, 12, -22]],
                "stable_weight": 202,
                "front_1": -52,
                "front_2": -27,
                "num_1": -44,
                "num_2": -51}


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


class AI(object):
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        self.color = color
        self.time_out = time_out
        self.candidate_list = []
        self.game = Game(self.chessboard_size)

    @timeout_decorator.timeout(4.8)
    def go(self, chessboard):
        try:
            self.candidate_list.clear()
            left_number = len(np.where(chessboard == COLOR_NONE)[0])
            if left_number <= 10:
                legal_moves = self.game.get_legal_moves(chessboard, self.color)
                if len(legal_moves) == 0:
                    return
                self.candidate_list.extend(legal_moves)
                player = MinimaxSearchPlayer(self.game, chessboard, self.color, 2, params)
                best_action = player.get_action()
                self.candidate_list.append(best_action)
                player = FinalSearch(self.game, self.color)
                player.search(chessboard, self.color, 0)
                if player.best_move is not None:
                    self.candidate_list.append(player.best_move)

            else:
                legal_moves = self.game.get_legal_moves(chessboard, self.color)
                if len(legal_moves) == 0:
                    return
                self.candidate_list.extend(legal_moves)
                search_depth = 3
                while True:
                    if search_depth >= 20:
                        break
                    player = MinimaxSearchPlayer(self.game, chessboard, self.color, search_depth, params)
                    move = player.get_action()
                    self.candidate_list.append(move)
                    search_depth += 1
        except:
            print()

if __name__ == '__main__':
    l = params["action_matrix_2"]
    for i in range(8):
        for j in range(8):
            print(l[i][j], end='')
            if j < 7:
                print('& ', end='')
            else:
                print('\\\\', end='')
        print()

