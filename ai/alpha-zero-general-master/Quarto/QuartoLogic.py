class Board():
    def __init__(self):
        "Set up initial board configuration."
        self.board = np.zeros((4, 4), dtype=int)
        self.pieces = self.create_pieces()

    def create_pieces(self):
        pieces = []
        for color in ['L', 'D']:
            for shape in ['S', 'C']:
                for height in ['S', 'T']:
                    for hole in ['S', 'H']:
                        pieces.append(Piece(color, shape, height, hole))
        return pieces

    def place_piece(self, row, col, piece):
        self.board[row][col] = piece

    def get_empty_positions(self):
        empty_positions = []
        for row in range(4):
            for col in range(4):
                if self.board[row][col] == 0:
                    empty_positions.append((row, col))
        return empty_positions

    def check_win(self):
        # Check rows, columns and diagonals for a win
        for i in range(4):
            if self.check_line(self.board[i, :]) or self.check_line(self.board[:, i]):
                return True
        if self.check_line(self.board.diagonal()) or self.check_line(np.fliplr(self.board).diagonal()):
            return True
        return False

    def check_line(self, line):
        # Check if all elements in the line are the same and not empty
        if np.all(line == line[0]) and line[0] != 0:
            return True
        return False