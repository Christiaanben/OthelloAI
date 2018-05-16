import numpy as np
import random

'''
1's represent white tiles
-1's represent black tiles
'''


class AI:
    def __init__(self, player=None, name=None, n_hidden_nodes=100, weight1=None, weight2=None):
        if weight1 is None:
            self.weight1 = np.zeros((n_hidden_nodes, 2*64))
            for i in range(len(self.weight1)):
                for j in range(len(self.weight1[i])):
                    self.weight1[i, j] = random.random()*2 - 1
        else:
            self.weight1 = weight1
        if weight2 is None:
            self.weight2 = np.zeros((64, n_hidden_nodes+1))
            for i in range(len(self.weight2)):
                for j in range(len(self.weight2[i])):
                    self.weight2[i, j] = random.random()*2 - 1
        else:
            self.weight2 = weight2
        self.player = player
        self.name = name

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name

    def set_player(self, player):
        self.player = player

    def get_player(self):
        return self.player

    def get_weights(self):
        return self.weight1, self.weight2

    def set_weights(self, weight1, weight2):
        self.weight1 = weight1
        self.weight2 = weight2

    def get_next_move(self, board, available_moves):
        board = (self.player*board)
        n_rotations = np.argmax(Othello.quadrant_value(board))
        rotated_board = Othello.rotate(board, n_rotations)
        rotated_moves = Othello.rotate(available_moves, n_rotations)
        input_layer = np.reshape(rotated_board, 64)
        input_layer = np.append(input_layer, np.reshape(rotated_moves, 64))
        x = self.sigmoid(np.dot(self.weight1, input_layer))
        x = np.append(x, 1)
        y = self.sigmoid(np.dot(self.weight2, x))
        rotated_output = np.reshape(y, (8, 8))
        output = Othello.rotate(rotated_output, (4-n_rotations) % 4)
        index = np.unravel_index(output.argmax(), output.shape)
        return index

    @staticmethod
    def sigmoid(nodes):
        for i in range(len(nodes)):
            nodes[i] = 1/(1+np.exp(nodes[i]))
        return nodes

    def mutate(self, chance=0.01):
        # Mutations occur on intervals of 0.4%
        # Mutations:
        # 1. Replace weight with completely new weight
        # 2. Scale it by some random factor
        # 3. add a random number
        # 4. Inverse weight
        weight1 = self.weight1
        weight2 = self.weight2
        for i in range(len(weight1)):
            for j in range(len(weight1[i])):
                prob = random.random()
                if prob < chance*1/4:
                    weight1[i, j] = 2*random.random() - 1
                elif prob < chance*2/4:
                    weight1[i, j] = weight1[i, j] * (2 * random.random() - 1)
                elif prob < chance*3/4:
                    weight1[i, j] = weight1[i, j] + random.random() - 0.5
                elif prob < chance:
                    weight1[i, j] = -1 * weight1[i, j]
                weight1[i, j] = max(-10.0, min(10.0, weight1[i, j]))
        for i in range(len(weight2)):
            for j in range(len(weight2[i])):
                prob = random.random()
                if prob < 0.005:
                    weight2[i, j] = 2*random.random() - 1
                elif prob < 0.01:
                    weight2[i, j] = weight2[i, j] * (2 * random.random() - 1)
                elif prob < 0.015:
                    weight2[i, j] = weight2[i, j] + random.random() - 0.5
                elif prob < 0.020:
                    weight2[i, j] = -1 * weight2[i, j]
                weight2[i, j] = max(-10.0, min(10.0, weight2[i, j]))
        return weight1, weight2


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

    @staticmethod
    def rotate(board, n_rotations):
        for n in range(n_rotations):
            rotated = zip(*board[::-1])
            board = np.array(list(rotated))
        return board

    @staticmethod
    def quadrant_value(board):
        '''
        Quadrants look like this:
        0|3
        1|2
        so that i can rotate easily
        :param board:
        :return:
        '''
        q = [[], [], [], []]
        for i in range(8):
            for j in range(8):
                row, col = int(i / 4), int(j / 4)
                index = int(np.abs(row - 3*col))
                q[index].append(board[i, j])
        value = np.zeros(4)
        for i in range(4):
            unique, counts = np.unique(q[i], return_counts=True)
            dictionary = dict(zip(unique, counts))
            if 1 in dictionary:
                value[i] = dictionary[1]
        return value

    def play(self, ai, player=1):
        ai.set_player(-player)

        if player == -1:
            if self.any_move(player):
                self.display()
                row, col = list(map(int, input('Input:').split(',')))
                print('Player put a '+str(player)+' at ('+str(row)+','+str(col)+')')
                if not self.check_move(row, col, player):
                    print('AI wins')
                    return
                self.make_move(row, col, player)

        moved = True
        while moved:
            moved = False
            available, moves = self.any_move(ai.get_player())
            if available:
                self.display()
                row, col = ai.get_next_move(self.board, moves)
                print('AI put a '+str(ai.get_player())+' at ('+str(row)+','+str(col)+')')
                if not self.check_move(row, col, ai.get_player())[0]:
                    print('Player wins')
                    return
                moved = True
                self.make_move(row, col, ai.get_player())
            if self.any_move(player):
                self.display()
                row, col = list(map(int, input('Input:').split(',')))
                print('Player put a '+str(player)+' at ('+str(row)+','+str(col)+')')
                if not self.check_move(row, col, player)[0]:
                    print('AI wins')
                    return
                self.make_move(row, col, player)
        result = self.get_result()
        if result[player] >= result[ai.get_player()]:
            print('Player wins')
        else:
            print('AI wins')

    def display(self):
        print(self.board)

    def play_self(self, player1, player2, display_board = False):
        black = player2
        white = player1
        if random.random() < 0.5:
            black = player1
            white = player2
        black.set_player(-1)
        white.set_player(1)

        moved = True
        while moved:
            moved = False
            available, moves = self.any_move(black.get_player())
            if available:
                row, col = black.get_next_move(self.board, moves)
                if display_board:
                    self.display()
                    print('AI put a ' + str(black.get_player()) + ' at (' + str(row) + ',' + str(col) + ')')
                if not self.check_move(row, col, black.get_player())[0]:
                    return white
                moved = True
                self.make_move(row, col, black.get_player())
            available, moves = self.any_move(white.get_player())
            if available:
                row, col = white.get_next_move(self.board, moves)
                if display_board:
                    self.display()
                    print('AI put a ' + str(white.get_player()) + ' at (' + str(row) + ',' + str(col) + ')')
                if not self.check_move(row, col, white.get_player())[0]:
                    return black
                moved = True
                self.make_move(row, col, white.get_player())
        result = self.get_result()
        if result[-1] > result[1]:
            return black
        return white

    def get_result(self):
        unique, counts = np.unique(self.board, return_counts=True)
        return dict(zip(unique, counts))

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
        if not opposite_player_adjacent:
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
                if not (-1 < tile[0] < 8 and -1 < tile[1] < 8):
                    tile = np.subtract(tile, delta)
                    break
            if self.board[tile[0], tile[1]] == player:
                valid = True
        if not valid:
            return False, "Error - line is not closed on both ends by the player's tile"

        return True, 'Success'

    def any_move(self, player):
        available = False
        moves = np.zeros((8, 8))
        for i in range(8):
            for j in range(8):
                if self.check_move(i, j, player)[0]:
                    available = True
                    moves[i, j] = 1
        return available, moves
