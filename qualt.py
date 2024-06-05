#ゲームシステム
class QuartoGame:
    def __init__(self):
        self.board = [[None for _ in range(4)] for _ in range(4)]
        self.pieces = [format(i, '04b') for i in range(16)]
        self.available_pieces = set(self.pieces)

    def display_board(self):
        for row in self.board:
            print(" | ".join(piece if piece is not None else "----" for piece in row))
            print("-" * 29)

    def is_winning_move(self):
        for i in range(4):
            if self.check_line([self.board[i][j] for j in range(4)]):
                return True
            if self.check_line([self.board[j][i] for j in range(4)]):
                return True
        if self.check_line([self.board[i][i] for i in range(4)]):
            return True
        if self.check_line([self.board[i][3 - i] for i in range(4)]):
            return True
        return False

    def check_line(self, line):
        if None in line:
            return False
        attributes = [0, 0, 0, 0]
        for piece in line:
            for i in range(4):
                attributes[i] |= (int(piece[i]) << i)
        return any(attr == 15 or attr == 0 for attr in attributes)

    def make_move(self, row, col, piece):
        self.board[row][col] = piece
        self.available_pieces.remove(piece)

    def get_available_pieces(self):
        return list(self.available_pieces)

    def get_available_moves(self):
        moves = []
        for i in range(4):
            for j in range(4):
                if self.board[i][j] is None:
                    moves.append((i, j))
        return moves


#プレイヤー処理
def get_player_move(game):
    available_moves = game.get_available_moves()
    move = None
    while move not in available_moves:
        try:
            row = int(input("Choose row (0-3): "))
            col = int(input("Choose column (0-3): "))
            move = (row, col)
            if move not in available_moves:
                print("Invalid move. Try again.")
        except ValueError:
            print("Invalid input. Please enter numbers between 0 and 3.")
    return move

def get_player_piece(game):
    available_pieces = game.get_available_pieces()
    piece = None
    while piece not in available_pieces:
        piece = input(f"Choose a piece from {available_pieces}: ")
        if piece not in available_pieces:
            print("Invalid piece. Try again.")
    return piece

#ゲーム実行
def play_game():
    game = QuartoGame()
    current_player = 1

    while True:
        print(f"Player {current_player}'s turn")
        game.display_board()

        if not game.get_available_moves():
            print("It's a draw!")
            break

        if game.is_winning_move():
            print(f"Player {3 - current_player} wins!")
            break

        piece_to_place = get_player_piece(game)
        move = get_player_move(game)
        game.make_move(move[0], move[1], piece_to_place)

        if game.is_winning_move():
            game.display_board()
            print(f"Player {current_player} wins!")
            break

        current_player = 3 - current_player

if __name__ == "__main__":
    play_game()
    #ari
