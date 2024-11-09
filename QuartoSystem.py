import numpy as np
import random

# ゲームに必要なクラス
class Piece:
    def __init__(self, color, shape, height, hole):
        self.color = color
        self.shape = shape
        self.height = height
        self.hole = hole

    def __repr__(self):
        return f"{self.color}{self.shape}{self.height}{self.hole}"

class Board:
    def __init__(self):
        self.board = [[None for _ in range(4)] for _ in range(4)]

    def place_piece(self, row, col, piece):
        self.board[row][col] = piece

    def check_win(self):
        for i in range(4):
            if self.check_line([self.board[i][j] for j in range(4)]) or \
               self.check_line([self.board[j][i] for j in range(4)]):
                return True
        return self.check_line([self.board[i][i] for i in range(4)]) or \
               self.check_line([self.board[i][3 - i] for i in range(4)])

    def check_line(self, line):
        for attr in ['color', 'shape', 'height', 'hole']:
            values = [getattr(piece, attr) for piece in line if piece is not None]
            if len(values) == 4 and all(v == values[0] for v in values):
                return True
        return False

class QuartoGame:
    def __init__(self):
        self.board = Board()
        self.pieces = self.create_pieces()
        self.current_piece = None

    def create_pieces(self):
        pieces = []
        for color in ['L', 'D']:
            for shape in ['S', 'C']:
                for height in ['S', 'T']:
                    for hole in ['S', 'H']:
                        pieces.append(Piece(color, shape, height, hole))
        return pieces

    def select_piece_for_opponent(self):
        if self.pieces:
            piece = random.choice(self.pieces)
            self.pieces.remove(piece)
            return piece
        return None

    def select_piece_from_remaining(self, index):
        if 0 <= index < len(self.pieces):
            piece = self.pieces[index]
            del self.pieces[index]
            return piece
        return None

    def get_state(self):
        state = []
        for row in self.board.board:
            state += [1 if piece else 0 for piece in row]
        state += [1 if piece else 0 for piece in self.pieces]
        return np.array(state, dtype=np.float32)