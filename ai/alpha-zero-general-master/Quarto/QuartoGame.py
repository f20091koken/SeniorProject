import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Game import Game
from .QuartoLogic import Board

class QuartoGame(Game):
    def __init__(self):
        self.board = Board()

    def getInitBoard(self):
        return self.board.getInitBoard()

    def getBoardSize(self):
        return self.board.getBoardSize()

    def getActionSize(self):
        return self.board.getActionSize()

    def getNextState(self, board, player, action, piece):
        return self.board.getNextState(board, player, action, piece)

    def getValidMoves(self, board, player):
        return self.board.getValidMoves(board, player)

    def getGameEnded(self, board, player):
        return self.board.getGameEnded(board, player)

    def getCanonicalForm(self, board, player):
        return self.board.getCanonicalForm(board, player)

    def getSymmetries(self, board, pi):
        return self.board.getSymmetries(board, pi)

    def stringRepresentation(self, board):
        return self.board.stringRepresentation(board)