# import random

# class QuartoGame:
#     def __init__(self):
#         self.board = [[None for _ in range(4)] for _ in range(4)]
#         self.pieces = self.initialize_pieces()
#         self.available_pieces = set(self.pieces.keys())
    
#     def initialize_pieces(self):
#         pieces = {}
#         piece_id = 0
#         for color in [0, 1]:  # 0: Light, 1: Dark
#             for shape in [0, 1]:  # 0: Square, 1: Circle
#                 for height in [0, 1]:  # 0: Short, 1: Tall
#                     for surface in [0, 1]:  # 0: Solid, 1: Hollow
#                         pieces[piece_id] = (color, shape, height, surface)
#                         piece_id += 1
#         return pieces

#     def piece_to_string(self, piece):
#         if piece is None:
#             return "    "
#         color, shape, height, surface = piece
#         color_str = "L" if color == 0 else "D"
#         shape_str = "S" if shape == 0 else "C"
#         height_str = "S" if height == 0 else "T"
#         surface_str = "S" if surface == 0 else "H"
#         return f"{color_str}{shape_str}{height_str}{surface_str}"

#     def display_board(self):
#         print("    0     1     2     3")
#         print("  +-----+-----+-----+-----+")
#         for i, row in enumerate(self.board):
#             row_str = " | ".join([self.piece_to_string(cell) for cell in row])
#             print(f"{i} | {row_str} |")
#             print("  +-----+-----+-----+-----+")

#     def is_winner(self):
#         # Check rows, columns and diagonals for a winning condition
#         for i in range(4):
#             if self.check_line([self.board[i][j] for j in range(4)]) or \
#                self.check_line([self.board[j][i] for j in range(4)]):
#                 return True
#         if self.check_line([self.board[i][i] for i in range(4)]) or \
#            self.check_line([self.board[i][3-i] for i in range(4)]):
#             return True
#         return False

#     def check_line(self, line):
#         if None in line:
#             return False
#         for i in range(4):
#             if all(p[i] == line[0][i] for p in line):
#                 return True
#         return False

#     def make_move(self, position, piece):
#         x, y = position
#         self.board[x][y] = self.pieces[piece]
#         self.available_pieces.remove(piece)

#     def get_best_move(self, piece):
#         # Simple AI that selects a random valid move
#         valid_moves = [(x, y) for x in range(4) for y in range(4) if self.board[x][y] is None]
#         return random.choice(valid_moves)

#     def available_pieces_to_string(self):
#         return {self.piece_to_string(self.pieces[piece]) for piece in self.available_pieces}

#     def string_to_piece(self, piece_str):
#         for piece, attributes in self.pieces.items():
#             if self.piece_to_string(attributes) == piece_str:
#                 return piece
#         return None

#     def play(self):
#         current_player = "human"
#         while True:
#             self.display_board()
#             available_piece_strings = self.available_pieces_to_string()
#             if current_player == "human":
#                 piece_str = input(f"Select a piece from available pieces {available_piece_strings}: ")
#                 piece = self.string_to_piece(piece_str)
#                 x = int(input("Select row (0-3): "))
#                 y = int(input("Select column (0-3): "))
#                 self.make_move((x, y), piece)
#                 if self.is_winner():
#                     self.display_board()
#                     print("Human wins!")
#                     break
#                 current_player = "AI"
#             else:
#                 piece = random.choice(list(self.available_pieces))
#                 move = self.get_best_move(piece)
#                 self.make_move(move, piece)
#                 print(f"AI placed piece {self.piece_to_string(self.pieces[piece])} at position {move}")
#                 if self.is_winner():
#                     self.display_board()
#                     print("AI wins!")
#                     break
#                 current_player = "human"

# if __name__ == "__main__":
#     game = QuartoGame()
#     game.play()


import random
import math

class QuartoGame:
    def __init__(self):
        self.board = [[None for _ in range(4)] for _ in range(4)]
        self.pieces = self.initialize_pieces()
        self.available_pieces = set(self.pieces.keys())

    def initialize_pieces(self):
        pieces = {}
        piece_id = 0
        for color in [0, 1]:  # 0: Light, 1: Dark
            for shape in [0, 1]:  # 0: Square, 1: Circle
                for height in [0, 1]:  # 0: Short, 1: Tall
                    for surface in [0, 1]:  # 0: Solid, 1: Hollow
                        pieces[piece_id] = (color, shape, height, surface)
                        piece_id += 1
        return pieces

    def piece_to_string(self, piece):
        if piece is None:
            return "    "
        color, shape, height, surface = piece
        color_str = "L" if color == 0 else "D"
        shape_str = "S" if shape == 0 else "C"
        height_str = "S" if height == 0 else "T"
        surface_str = "S" if surface == 0 else "H"
        return f"{color_str}{shape_str}{height_str}{surface_str}"

    def display_board(self):
        print("    0     1     2     3")
        print("  +-----+-----+-----+-----+")
        for i, row in enumerate(self.board):
            row_str = " | ".join([self.piece_to_string(cell) for cell in row])
            print(f"{i} | {row_str} |")
            print("  +-----+-----+-----+-----+")

    def is_winner(self):
        # Check rows, columns and diagonals for a winning condition
        for i in range(4):
            if self.check_line([self.board[i][j] for j in range(4)]) or \
               self.check_line([self.board[j][i] for j in range(4)]):
                return True
        if self.check_line([self.board[i][i] for i in range(4)]) or \
           self.check_line([self.board[i][3-i] for i in range(4)]):
            return True
        return False

    def check_line(self, line):
        if None in line:
            return False
        for i in range(4):
            if all(p[i] == line[0][i] for p in line):
                return True
        return False

    def make_move(self, position, piece):
        x, y = position
        self.board[x][y] = self.pieces[piece]
        self.available_pieces.remove(piece)

    def get_best_move(self, piece):
        best_score = -math.inf
        best_move = None
        for move in [(x, y) for x in range(4) for y in range(4) if self.board[x][y] is None]:
            self.make_move(move, piece)
            score = self.minimax(0, False, -math.inf, math.inf)
            self.undo_move(move, piece)
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def minimax(self, depth, is_maximizing, alpha, beta, max_depth=3):
        if depth >= max_depth or self.is_winner():
            return self.evaluate_board(is_maximizing)
        if not any(None in row for row in self.board):
            return 0

        if is_maximizing:
            max_eval = -math.inf
            for move in [(x, y) for x in range(4) for y in range(4) if self.board[x][y] is None]:
                for piece in self.available_pieces:
                    self.make_move(move, piece)
                    eval = self.minimax(depth + 1, False, alpha, beta, max_depth)
                    self.undo_move(move, piece)
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
            return max_eval
        else:
            min_eval = math.inf
            for move in [(x, y) for x in range(4) for y in range(4) if self.board[x][y] is None]:
                for piece in self.available_pieces:
                    self.make_move(move, piece)
                    eval = self.minimax(depth + 1, True, alpha, beta, max_depth)
                    self.undo_move(move, piece)
                    min_eval = min(min_eval, eval)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
            return min_eval

    def undo_move(self, position, piece):
        x, y = position
        self.board[x][y] = None
        self.available_pieces.add(piece)

    def available_pieces_to_string(self):
        return {self.piece_to_string(self.pieces[piece]) for piece in self.available_pieces}

    def string_to_piece(self, piece_str):
        for piece, attributes in self.pieces.items():
            if self.piece_to_string(attributes) == piece_str:
                return piece
        return None

    def evaluate_board(self, is_maximizing):
        if self.is_winner():
            return 1 if is_maximizing else -1
        return 0

    def get_best_piece(self):
        best_score = -math.inf
        best_piece = None
        for piece in self.available_pieces:
            score = self.evaluate_piece(piece)
            if score > best_score:
                best_score = score
                best_piece = piece
        return best_piece

    def evaluate_piece(self, piece):
        # Simple heuristic for piece evaluation
        # Can be customized to be more sophisticated
        return random.random()

    def play(self):
        current_player = "human"
        selected_piece = None

        while True:
            self.display_board()
            available_piece_strings = self.available_pieces_to_string()
            if current_player == "human":
                if selected_piece is not None:
                    x = int(input("Select row (0-3): "))
                    y = int(input("Select column (0-3): "))
                    self.make_move((x, y), selected_piece)
                    if self.is_winner():
                        self.display_board()
                        print("Human wins!")
                        break
                selected_piece_str = input(f"Select a piece for AI from available pieces {available_piece_strings}: ")
                selected_piece = self.string_to_piece(selected_piece_str)
                current_player = "AI"
            else:
                move = self.get_best_move(selected_piece)
                self.make_move(move, selected_piece)
                print(f"AI placed piece {self.piece_to_string(self.pieces[selected_piece])} at position {move}")
                if self.is_winner():
                    self.display_board()
                    print("AI wins!")
                    break
                selected_piece = self.get_best_piece()  # AI selects a piece for the human
                print(f"AI selected piece {self.piece_to_string(self.pieces[selected_piece])} for human to place")
                current_player = "human"

if __name__ == "__main__":
    game = QuartoGame()
    game.play()
