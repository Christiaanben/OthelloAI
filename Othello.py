import numpy as np
import random

'''
1's represent white tiles
-1's represent black tiles
'''


class AI:

    def __init__(self, player, n_hidden_nodes=16):
        self.weight1 = np.zeros((n_hidden_nodes-1, 64))
        for i in range(len(self.weight1)):
            for j in range(len(self.weight1[i])):
                self.weight1[i, j] = random.random()
        self.weight2 = np.ones((n_hidden_nodes-1, n_hidden_nodes))
        self.weight3 = np.ones((64,n_hidden_nodes))
        self.player = player

    def get_next_move(self, board):
        input_layer = np.reshape(board, 64)
        x = self.sigmoid(np.dot(self.weight1, input_layer))
        x = np.append(x, 1)
        y = self.sigmoid(np.dot(self.weight2, x))
        y = np.append(y, 1)
        z = self.sigmoid(np.dot(self.weight3, y))
        return np.argmax(z)

    def sigmoid(self, nodes):
        for i in range(len(nodes)):
            nodes[i] = 1/(1+np.exp(nodes[i]))
        return nodes

class Othello:
    board_default = board1 = [[0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, -1, 0, 0, 0],
                              [0, 0, 0, -1, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0]]

    def __init__(self, board=board_default):
        self.board = np.zeros((8, 8))
        for i in range(8):
            for j in range(8):
                self.board[i][j] = board[i][j]

    def get_board(self):
        return self.board

    def make_move(self, row, col, player):
        # Check if tile is already taken:
        if self.board[row, col] != 0:
            return 'Error - Tile taken'

        # Check if tile has adjacent tile:
        adjacent_tile = []
        opposite_player_adjacent = False
        for i in range(9):
            try:
                row_to_check = int(row - 1 + i / 3)
                col_to_check = int(col - 1 + i % 3)
                if self.board[row_to_check, col_to_check] == (-player):
                    adjacent_tile.append([row_to_check,col_to_check])
                    opposite_player_adjacent = True
            except:
                pass
        if not(opposite_player_adjacent):
            return "Error - Tile not adjacent to the opponent's tile"

        # Check if adjacent tile closes on both ends
        valid = False
        for tile in adjacent_tile:
            delta = [tile[0]-row, tile[1]-col]
            while self.board[tile[0], tile[1]] != player:
                if self.board[tile[0], tile[1]] == 0:
                    break
                tile = np.add(tile, delta)
            if self.board[tile[0], tile[1]] == player:
                valid = True
                tile = np.subtract(tile, delta)
                while tile[0] != row or tile[1] != col:
                    self.board[tile[0], tile[1]] = player
                    tile = np.subtract(tile, delta)
        if not valid:
            return "Error - line is not closed on both ends by the player's tile"

        self.board[row, col] = player
        return 'Success'

    # Check to see if the given player can play a tile at the given location
    def check_move(self, row, col, player):
        # Check if tile is already taken:
        if self.board[row, col] != 0:
            return False, 'Error - Tile taken'

        # Check if tile has adjacent tile:
        adjacent_tile = []
        opposite_player_adjacent = False
        for i in range(9):
            try:
                row_to_check = int(row - 1 + i / 3)
                col_to_check = int(col - 1 + i % 3)
                if self.board[row_to_check, col_to_check] == (-player):
                    adjacent_tile.append([row_to_check, col_to_check])
                    opposite_player_adjacent = True
            except:
                pass
        if not opposite_player_adjacent:
            return False, "Error - Tile not adjacent to the opponent's tile"

        # Check if adjacent tile closes on both ends
        valid = False
        for tile in adjacent_tile:
            delta = [tile[0] - row, tile[1] - col]
            while self.board[tile[0], tile[1]] != player:
                if self.board[tile[0], tile[1]] == 0:
                    break
                tile = np.add(tile, delta)
            if self.board[tile[0], tile[1]] == player:
                valid = True
        if not valid:
            return False, "Error - line is not closed on both ends by the player's tile"

        return True, 'Success'

    def any_move(self, player):
        for i in range(8):
            for j in range(8):
                if self.check_move(i, j, player)[0]:
                    return True
        return False
