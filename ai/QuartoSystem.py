import numpy as np
import random

# ゲームに必要なクラス
class Piece:
    def __init__(self, color, shape, height, hole):
        self.color = color
        self.shape = shape
        self.height = height
        self.hole = hole

    def attributes(self):
        # 属性をまとめてリストで返す
        return [self.color, self.shape, self.height, self.hole]

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
    
    def get_state(self):
    # ボードの状態を1次元のnumpy配列として取得
        state = []
        for row in self.board:
            for cell in row:
                state.append(1 if cell is not None else 0)
        return np.array(state, dtype=np.float32)

class QuartoGame:
    def __init__(self):
        self.board = Board()
        self.pieces = self.create_pieces()
        self.current_piece = None
        self.reset_game()

    def reset_game(self):
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
    
    def check_draw(self):
        # 残りのピースがなくなった場合の判定
        return len(self.pieces) == 0 and not self.board.check_win()

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