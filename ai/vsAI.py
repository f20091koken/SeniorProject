import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from qualt import Piece, Board, QuartoGame, DQNAgent


class QuartoGamePlay():
    def __init__(self, agent, human_first=True):
        self.game = QuartoGame()
        self.agent = agent
        self.human_first = human_first

    def start_game(self):
        self.game.reset()
        is_human_turn = self.human_first

        while True:
            state = np.reshape(self.game.get_encoded_state(), [1, 4 * 4 * 4])

            if is_human_turn:
                print("Your turn. Enter row and column (0-3 each) separated by space:")
                row, col = map(int, input().split())
                if (row, col) not in self.game.board.get_empty_positions():
                    print("Invalid position. Try again.")
                    continue
                piece = random.choice(self.game.pieces)
                self.game.board.place_piece(row, col, piece)

            else:
                action = self.agent.act(state)
                row, col = divmod(action, 4)
                piece = random.choice(self.game.pieces)
                print(f"AI places piece at ({row}, {col})")
                self.game.board.place_piece(row, col, piece)

            if self.game.board.check_win():
                print("You win!" if is_human_turn else "AI wins!")
                break

            if not self.game.board.get_empty_positions():
                print("It's a draw!")
                break

            is_human_turn = not is_human_turn

if __name__ == "__main__":
    state_size = 4 * 4 * 4
    action_size = 16
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    
    # 保存済みモデルのロード
    agent.load_model("agent1_model.pth")
    
    # 人間 vs AI の対戦
    print("Starting a game against the AI!")

        # QuartoGamePlayクラスを使用して対戦を開始
    game_play = QuartoGamePlay(agent, human_first=True)  # human_first=False にするとAIが先攻
    game_play.start_game()  # 対戦を開始


