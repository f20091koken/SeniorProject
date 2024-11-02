# import random
# import math

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
#         best_score = -math.inf
#         best_move = None
#         for move in [(x, y) for x in range(4) for y in range(4) if self.board[x][y] is None]:
#             self.make_move(move, piece)
#             score = self.minimax(0, False, -math.inf, math.inf)
#             self.undo_move(move, piece)
#             if score > best_score:
#                 best_score = score
#                 best_move = move
#         return best_move

#     def minimax(self, depth, is_maximizing, alpha, beta, max_depth=3):
#         if depth >= max_depth or self.is_winner():
#             return self.evaluate_board(is_maximizing)
#         if not any(None in row for row in self.board):
#             return 0

#         if is_maximizing:
#             max_eval = -math.inf
#             for move in [(x, y) for x in range(4) for y in range(4) if self.board[x][y] is None]:
#                 for piece in self.available_pieces:
#                     self.make_move(move, piece)
#                     eval = self.minimax(depth + 1, False, alpha, beta, max_depth)
#                     self.undo_move(move, piece)
#                     max_eval = max(max_eval, eval)
#                     alpha = max(alpha, eval)
#                     if beta <= alpha:
#                         break
#             return max_eval
#         else:
#             min_eval = math.inf
#             for move in [(x, y) for x in range(4) for y in range(4) if self.board[x][y] is None]:
#                 for piece in self.available_pieces:
#                     self.make_move(move, piece)
#                     eval = self.minimax(depth + 1, True, alpha, beta, max_depth)
#                     self.undo_move(move, piece)
#                     min_eval = min(min_eval, eval)
#                     beta = min(beta, eval)
#                     if beta <= alpha:
#                         break
#             return min_eval

#     def undo_move(self, position, piece):
#         x, y = position
#         self.board[x][y] = None
#         self.available_pieces.add(piece)

#     def available_pieces_to_string(self):
#         return {self.piece_to_string(self.pieces[piece]) for piece in self.available_pieces}

#     def string_to_piece(self, piece_str):
#         for piece, attributes in self.pieces.items():
#             if self.piece_to_string(attributes) == piece_str:
#                 return piece
#         return None

#     def evaluate_board(self, is_maximizing):
#         if self.is_winner():
#             return 1 if is_maximizing else -1
#         return 0

#     def get_best_piece(self):
#         best_piece = None
#         best_piece_score = math.inf
#         for piece in self.available_pieces:
#             piece_score = self.simulate_best_move(piece)
#             if piece_score < best_piece_score:
#                 best_piece_score = piece_score
#                 best_piece = piece
#         return best_piece

#     def simulate_best_move(self, piece):
#         best_score = math.inf
#         for move in [(x, y) for x in range(4) for y in range(4) if self.board[x][y] is None]:
#             self.make_move(move, piece)
#             if self.is_winner():
#                 self.undo_move(move, piece)
#                 return -1
#             score = self.minimax(0, False, -math.inf, math.inf)
#             self.undo_move(move, piece)
#             best_score = min(best_score, score)
#         return best_score

#     def play(self):
#         current_player = "human"
#         selected_piece = None

#         while True:
#             self.display_board()
#             available_piece_strings = self.available_pieces_to_string()
#             if current_player == "human":
#                 if selected_piece is not None:
#                     x = int(input("Select row (0-3): "))
#                     y = int(input("Select column (0-3): "))
#                     self.make_move((x, y), selected_piece)
#                     if self.is_winner():
#                         self.display_board()
#                         print("Human wins!")
#                         break
#                 selected_piece_str = input(f"Select a piece for AI from available pieces {available_piece_strings}: ")
#                 selected_piece = self.string_to_piece(selected_piece_str)
#                 current_player = "AI"
#             else:
#                 move = self.get_best_move(selected_piece)
#                 self.make_move(move, selected_piece)
#                 print(f"AI placed piece {self.piece_to_string(self.pieces[selected_piece])} at position {move}")
#                 if self.is_winner():
#                     self.display_board()
#                     print("AI wins!")
#                     break
#                 selected_piece = self.get_best_piece()  # AI selects a piece for the human
#                 print(f"AI selected piece {self.piece_to_string(self.pieces[selected_piece])} for human to place")
#                 current_player = "human"

# if __name__ == "__main__":
#     game = QuartoGame()
#     game.play()



import random
import math
import csv
import numpy as np

class QuartoGame:
    def __init__(self):
        self.board = [[None for _ in range(4)] for _ in range(4)]
        self.pieces = self.initialize_pieces()
        self.available_pieces = set(self.pieces.keys())
        self.csv_file = 'quarto_moves.csv'
        self.q_table_file = 'q_table.csv'
        self.initialize_csv()

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

    def initialize_csv(self):
        with open(self.csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Piece', 'Row', 'Column'])

    def log_move(self, piece, position):
        piece_str = self.piece_to_string(self.pieces[piece])
        row, column = position
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([piece_str, row, column])

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
<<<<<<< HEAD
        best_piece = None
        best_piece_score = math.inf
        for piece in self.available_pieces:
            piece_score = self.simulate_best_move(piece)
            if piece_score < best_piece_score:
                best_piece_score = piece_score
                best_piece = piece
        return best_piece

    def simulate_best_move(self, piece):
        best_score = math.inf
        for move in [(x, y) for x in range(4) for y in range(4) if self.board[x][y] is None]:
            self.make_move(move, piece)
            if self.is_winner():
                self.undo_move(move, piece)
                return -1
            score = self.minimax(0, False, -math.inf, math.inf)
            self.undo_move(move, piece)
            best_score = min(best_score, score)
        return best_score

    def q_learning(self, alpha=0.1, gamma=0.9, epsilon=0.1, episodes=50000, eval_interval=1000, target_win_rate=0.9, eval_games=100):
        q_table = {}

        def get_q(state, action):
            return q_table.get((state, action), 0.0)

        for episode in range(episodes):
            self.__init__()
            state = self.get_state()
            while True:
                if random.uniform(0, 1) < epsilon:
                    piece = random.choice(list(self.available_pieces))
                    move = random.choice([(x, y) for x in range(4) for y in range(4) if self.board[x][y] is None])
                else:
                    piece, move = self.get_best_action(state, q_table)

                self.make_move(move, piece)
                next_state = self.get_state()
                reward = 1 if self.is_winner() else 0

                old_q = get_q(state, (piece, move))
                next_max_q = max([get_q(next_state, (next_piece, next_move)) for next_piece in self.available_pieces for next_move in [(x, y) for x in range(4) for y in range(4) if self.board[x][y] is None]], default=0)

                new_q = old_q + alpha * (reward + gamma * next_max_q - old_q)
                q_table[(state, (piece, move))] = new_q

                if self.is_winner() or not any(None in row for row in self.board):
                    break

                state = next_state

            if episode % eval_interval == 0:
                win_rate = self.evaluate_q_table(q_table, eval_games)
                print(f"Episode {episode}, Win Rate: {win_rate}")
                if win_rate >= target_win_rate:
                    break

        self.save_q_table_to_csv(q_table)

    def evaluate_q_table(self, q_table, games):
        wins = 0
        for _ in range(games):
            self.__init__()
            state = self.get_state()
            while True:
                piece, move = self.get_best_action(state, q_table)
                self.make_move(move, piece)
                if self.is_winner():
                    wins += 1
                    break
                if not any(None in row for row in self.board):
                    break
                state = self.get_state()
        return wins / games

    def get_best_action(self, state, q_table):
        best_q = -math.inf
        best_action = None
        for piece in self.available_pieces:
            for move in [(x, y) for x in range(4) for y in range(4) if self.board[x][y] is None]:
                q_value = q_table.get((state, (piece, move)), 0)
                if q_value > best_q:
                    best_q = q_value
                    best_action = (piece, move)
        if best_action is None:
            piece = random.choice(list(self.available_pieces))
            move = random.choice([(x, y) for x in range(4) for y in range(4) if self.board[x][y] is None])
            return piece, move
        return best_action

    def save_q_table_to_csv(self, q_table):
        with open(self.q_table_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['State', 'Piece', 'Move', 'Q-Value'])
            for (state, (piece, move)), q_value in q_table.items():
                writer.writerow([state, piece, move, q_value])

    def get_state(self):
        return tuple(tuple(row) for row in self.board)

    def play_ai_vs_ai(self, num_games=1):
        for _ in range(num_games):
            self.__init__()
            state = self.get_state()
            while True:
                piece, move = self.get_best_piece(), self.get_best_move(self.get_best_piece())
                self.make_move(move, piece)
                self.log_move(piece, move)
                if self.is_winner() or not any(None in row for row in self.board):
                    break
                state = self.get_state()
            self.display_board()
game = QuartoGame()
# game.q_learning()


game.play_ai_vs_ai()
=======
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
>>>>>>> fa851eec9fdf94ff133695998c39cb51e5d5952e

