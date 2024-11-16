import sys
sys.path.append('ai/alpha-zero-general-master')
from Game import Game
from .QuartoLogic import Board
import numpy as np

class QuartoGame(Game):
    def __init__(self):
        self.board = Board()
        self.pieces = self.create_pieces()

    def create_pieces(self):
        pieces = []
        for color in ['L', 'D']:
            for shape in ['S', 'C']:
                for height in ['S', 'T']:
                    for hole in ['S', 'H']:
                        pieces.append(Piece(color, shape, height, hole))
        return pieces

    def getInitBoard(self):
        # return initial board (numpy board)
        return np.array(self.board.board)

    def getBoardSize(self):
        # (a,b) tuple
        return (4, 4)

    def getActionSize(self):
        # return number of actions
        return 16

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        b = Board()
        b.board = np.copy(board)
        row, col = divmod(action, 4)
        b.place_piece(row, col, player)
        return (b.board, -player)

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        valids = [0] * self.getActionSize()
        b = Board()
        b.board = np.copy(board)
        empty_positions = b.get_empty_positions()
        for row, col in empty_positions:
            valids[row * 4 + col] = 1
        return np.array(valids)

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        b = Board()
        b.board = np.copy(board)
        if b.check_win():
            return 1 if player == 1 else -1
        return 0

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        return player * board

    def getSymmetries(self, board, pi):
        # mirror, rotational
        assert(len(pi) == self.getActionSize())
        pi_board = np.reshape(pi, (4, 4))
        l = []
        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()))]
        return l

    def stringRepresentation(self, board):
        return board.tostring()

    def stringRepresentationReadable(self, board):
        board_s = "".join(self.square_content[square] for row in board for square in row)
        return board_s

    def getScore(self, board, player):
        b = Board()
        b.board = np.copy(board)
        return b.check_win()

    @staticmethod
    def display(board):
        n = board.shape[0]
        print("   ", end="")
        for y in range(n):
            print(y, end=" ")
        print("")
        print("-----------------------")
        for y in range(n):
            print(y, "|", end="")    # print the row #
            for x in range(n):
                piece = board[y][x]    # get the piece to print
                print(QuartoGame.square_content[piece], end=" ")
            print("|")
        print("-----------------------")