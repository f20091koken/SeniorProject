import numpy as np

class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a] != 1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanQuartoPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        self.display(board)
        valid = self.game.getValidMoves(board, 1)
        while True:
            a = input("Enter your move: ")
            x, y = map(int, a.split())
            if valid[x * 4 + y]:
                break
            else:
                print('Invalid move')
        return x * 4 + y

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
                print(piece, end=" ")
            print("|")
        print("-----------------------")