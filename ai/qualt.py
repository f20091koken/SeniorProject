import numpy as np
import random
import tkinter as tk
from tkinter import messagebox, Toplevel, Listbox, Button
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

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
        # 縦・横・斜めで特徴が揃っているかをチェック
        for i in range(4):
            if self.check_line([self.board[i][j] for j in range(4)]) or \
               self.check_line([self.board[j][i] for j in range(4)]):
                return True
        return self.check_line([self.board[i][i] for i in range(4)]) or \
               self.check_line([self.board[i][3 - i] for i in range(4)])

    def check_line(self, line):
        """ 特徴が揃っているかを確認 """
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
        """残っているコマからインデックスでコマを選び、リストから削除します"""
        if 0 <= index < len(self.pieces):
            piece = self.pieces[index]
            del self.pieces[index]  # 選択したコマをリストから削除
            return piece
        return None

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model(torch.FloatTensor(state))
        return torch.argmax(act_values).item()

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          torch.max(self.target_model(torch.FloatTensor(next_state))).item())
            target_f = self.model(torch.FloatTensor(state))
            target_f[action] = target
            self.model.zero_grad()
            loss = nn.MSELoss()(target_f, self.model(torch.FloatTensor(state)))
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class QuartoGUI:
    def __init__(self, game, agent):
        self.game = game
        self.agent = agent
        self.root = tk.Tk()
        self.root.title("Quarto Game")
        self.create_board()
        self.player_turn = True  # プレイヤーのターンから始める
        self.next_piece = self.ai_selects_piece_for_player()  # AIが最初にプレイヤーのコマを選ぶ
        self.show_starting_piece()  # 最初のコマを表示

    def create_board(self):
        self.buttons = [[None for _ in range(4)] for _ in range(4)]
        for row in range(4):
            for col in range(4):
                button = tk.Button(self.root, text=" ", width=10, height=5,
                                   command=lambda r=row, c=col: self.on_click(r, c))
                button.grid(row=row, column=col)
                self.buttons[row][col] = button

    def show_starting_piece(self):
        messagebox.showinfo("Starting Piece", f"Your starting piece is: {self.next_piece}")

    def on_click(self, row, col):
        if self.game.board.board[row][col] is None and self.player_turn:
            # プレイヤーが選んだ位置にコマを置く
            self.game.board.place_piece(row, col, self.next_piece)
            self.buttons[row][col]["text"] = str(self.next_piece)
            
            if self.game.board.check_win():
                messagebox.showinfo("Game Over", "You Win!")
                self.root.quit()
            else:
                self.player_turn = False
                self.agent_turn()  # AIのターンを開始

    def ai_selects_piece_for_player(self):
        """AIがプレイヤーのために次に置くコマを選ぶ"""
        piece = self.game.select_piece_for_opponent()
        return piece

    def player_selects_piece_for_ai(self):
        """プレイヤーがAIのために次のコマを選択するためのリストボックスを表示"""
        piece_window = Toplevel(self.root)
        piece_window.title("Select a piece for AI")
        
        listbox = Listbox(piece_window, selectmode="single")
        for i, piece in enumerate(self.game.pieces):
            listbox.insert(i, str(piece))
        listbox.pack()

        def on_select():
            selected_index = listbox.curselection()
            if selected_index:
                piece_index = selected_index[0]
                piece = self.game.select_piece_from_remaining(piece_index)
                piece_window.destroy()  # 選択後ウィンドウを閉じる
                self.next_piece = piece  # 選択したコマを次に使うコマに設定

                # AIが選択したコマを配置する
                self.place_ai_piece()

        Button(piece_window, text="Select", command=on_select).pack()

    def place_ai_piece(self):
        # AIのターンで、選択したコマを盤面に配置
        row, col = self.ai_decides_position()  # AIが置く位置を決定
        if self.game.board.board[row][col] is None:
            self.game.board.place_piece(row, col, self.next_piece)
            self.buttons[row][col]["text"] = str(self.next_piece)

            if self.game.board.check_win():
                messagebox.showinfo("Game Over", "AI Wins!")
                self.root.quit()
            else:
                self.player_turn = True  # プレイヤーのターンに戻す
                self.next_piece = self.ai_selects_piece_for_player()  # 再度AIがプレイヤーのためのコマを選ぶ]
                messagebox.showinfo("AI's Turn", f"AI has selected the piece: {self.next_piece}")

    def ai_decides_position(self):
        """AIが配置する位置をランダムに決定する"""
        empty_positions = [(r, c) for r in range(4) for c in range(4) if self.game.board.board[r][c] is None]
        return random.choice(empty_positions) if empty_positions else (None, None)

    def agent_turn(self):
        """AIのターンを開始"""
        self.player_selects_piece_for_ai()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    game = QuartoGame()
    agent = DQNAgent(state_size=16, action_size=16)
    gui = QuartoGUI(game, agent)
    gui.run()