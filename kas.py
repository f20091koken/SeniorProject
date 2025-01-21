
#AlphaZeroChatGPT use

import numpy as np
import random
import json

# ボードの初期化
board = np.full((4, 4), None)  # Noneを未配置の状態に変更
pieces = [f"{i:04b}" for i in range(16)]  # コマの属性を2進数で表現
available_pieces = set(pieces)
game_log = []  # ログを保存するリスト

# コマを配置する関数
def place_piece(board, piece, row, col):
    if board[row, col] is not None:
        raise ValueError("Position already occupied!")
    board[row, col] = piece  # そのまま文字列で配置
    available_pieces.discard(piece)

# 勝利条件の判定
def check_win(board):
    for line in list(board) + list(board.T) + [board.diagonal(), np.fliplr(board).diagonal()]:
        if is_line_winning(line):
            return True
    return False

def is_line_winning(line):
    # None（未配置）のセルがある場合はスキップ
    if None in line:
        return False
    # 全セルの論理積または論理和で1属性でも一致するか確認
    values = [int(cell, 2) for cell in line]  # 2進数文字列を整数に変換
    return any(all((val >> i) & 1 for val in values) or 
               all(((~val) >> i) & 1 for val in values) 
               for i in range(4))

# ランダムAI
def random_ai(board, available_pieces):
    empty_positions = [(r, c) for r in range(4) for c in range(4) if board[r, c] is None]
    chosen_position = random.choice(empty_positions)  # ランダムに空きマスを選択
    return chosen_position

# ヒューリスティックAI（勝利手優先）
def heuristic_ai(board, piece):
    for row in range(4):
        for col in range(4):
            if board[row, col] is None:  # 空きマス
                temp_board = board.copy()
                temp_board[row, col] = piece
                if check_win(temp_board):
                    return (row, col)
    return random_ai(board, available_pieces)

# 人間プレイヤーの入力（相手に渡すコマを選ぶ）
def get_piece_choice(available_pieces):
    print("使用可能なコマ:", available_pieces)
    selected_piece = input("相手に渡すコマを選んでください: ")
    while selected_piece not in available_pieces:
        selected_piece = input("無効なコマです。再入力してください: ")
    return selected_piece

# 人間プレイヤーの入力（コマを設置する位置を選ぶ）
def get_position_choice(board):
    row, col = map(int, input("配置する位置（row col）を入力してください: ").split())
    while not (0 <= row < 4 and 0 <= col < 4 and board[row, col] is None):
        row, col = map(int, input("無効な位置です。再入力してください (row col): ").split())
    return (row, col)

# ボードの表示
def display_board(board):
    print("\n現在のボード状態:")
    for row in board:
        print(" | ".join(cell if cell is not None else "    " for cell in row))
        print("-" * 29)

# ログを記録
def log_turn(player, action, detail):
    game_log.append({
        "player": player,
        "action": action,
        "detail": detail
    })

# ゲームループ
def play_game():
    global board, available_pieces, game_log
    board = np.full((4, 4), None)  # ボードを初期化
    available_pieces = set(pieces)
    game_log = []

    # 先攻か後攻を選択
    while True:
        first_player = input("先攻か後攻を選んでください（先攻: 1, 後攻: 2）: ").strip()
        if first_player in {"1", "2"}:
            break
        print("無効な入力です。1または2を入力してください。")

    if first_player == "1":
        current_player = "Player"
        opponent = "AI"
    else:
        current_player = "AI"
        opponent = "Player"

    next_piece = None

    while True:
        display_board(board)
        print("使用可能なコマ:", available_pieces)

        if next_piece:
            print(f"{current_player}が受け取ったコマ: {next_piece}")

        if current_player == "Player":
            if next_piece:
                # プレイヤーがコマを設置
                print("プレイヤーのターン（コマを設置）")
                row, col = get_position_choice(board)
                place_piece(board, next_piece, row, col)
                log_turn(current_player, "place", {"piece": next_piece, "position": (row, col)})
            # プレイヤーが次のコマを選ぶ
            print("プレイヤーのターン（次のコマを選ぶ）")
            next_piece = get_piece_choice(available_pieces)
            log_turn(current_player, "choose", {"piece": next_piece})
        else:
            if next_piece:
                # AIがコマを設置
                print("AIのターン（コマを設置）")
                row, col = heuristic_ai(board, next_piece)
                place_piece(board, next_piece, row, col)
                log_turn(current_player, "place", {"piece": next_piece, "position": (row, col)})
            # AIが次のコマを選ぶ
            print("AIのターン（次のコマを選ぶ）")
            next_piece = random.choice(list(available_pieces))
            print(f"AIが選んだコマ: {next_piece}")
            log_turn(current_player, "choose", {"piece": next_piece})

        # 勝利判定
        if check_win(board):
            display_board(board)
            print(f"{current_player}の勝利!")
            break

        # 引き分け判定
        if not available_pieces:
            display_board(board)
            print("引き分け!")
            break

        # ターン交代
        current_player, opponent = opponent, current_player

    # ゲームログを保存
    with open("game_log.json", "w") as log_file:
        json.dump(game_log, log_file)
    print("ゲームログを保存しました: game_log.json")

# ゲームを開始
play_game()