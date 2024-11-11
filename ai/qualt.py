import numpy as np
import random
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

    def encode(self):
        """ピースの特徴を4ビットのベクトルとしてエンコード"""
        return [
            1 if self.color == 'D' else 0,
            1 if self.shape == 'C' else 0,
            1 if self.height == 'T' else 0,
            1 if self.hole == 'H' else 0
        ]

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

    def get_empty_positions(self):
        return [(row, col) for row in range(4) for col in range(4) if self.board[row][col] is None]

class QuartoGame:
    def __init__(self):
        self.board = Board()
        self.pieces = self.create_pieces()

    def create_pieces(self):
        pieces = []
        for color in ['L', 'D']:
            for shape in ['S', 'C']:
                for height in ['S', 'T']:
                    for hole in ['S', 'H']:
                        pieces.append(Piece(color, shape, height, hole))
        return pieces

    def reset(self):
        self.board = Board()
        self.pieces = self.create_pieces()

    def get_encoded_state(self):
        encoded_state = []
        for row in range(4):
            for col in range(4):
                piece = self.board.board[row][col]
                if piece is None:
                    encoded_state.extend([0, 0, 0, 0])
                else:
                    encoded_state.extend(piece.encode())
        return encoded_state

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
                # ターゲットネットワークからの出力をdetachで取り出す
                target = (reward + self.gamma *
                          torch.max(self.target_model(torch.FloatTensor(next_state))).detach().item())
            
            # Qネットワークの予測値
            target_f = self.model(torch.FloatTensor(state))
            
            # 行動サイズに合わせた出力の形状を確認
            target_f = target_f.view(-1)  # 1次元に変換
            
            # 正しいインデックスにターゲット値を設定
            target_f[action] = target
            
            # 損失計算とバックプロパゲーション
            self.model.zero_grad()
            output = self.model(torch.FloatTensor(state)).view(-1)
            loss = nn.MSELoss()(output, target_f)
            loss.backward()
            self.optimizer.step()
        
        # ε減少
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)
        torch.save(self.target_model.state_dict(), filepath + "_target")

    def load_model(self, filepath):
        self.model.load_state_dict(torch.load(filepath))
        self.target_model.load_state_dict(torch.load(filepath + "_target"))

    def train_agents(self, agent1, agent2, game, num_episodes=1000, batch_size=32, target_update_freq=10):
        for e in range(num_episodes):
            game.reset()
            state = game.get_encoded_state()
            state = np.reshape(state, [1, self.state_size])
            done = False
            agent1_turn = True

            while not done:
                agent = agent1 if agent1_turn else agent2
                action = agent.act(state)
                row, col = divmod(action, 4)
                piece = random.choice(game.pieces)  # ランダムなピースを選択
                game.board.place_piece(row, col, piece)

                if game.board.check_win():
                    reward1, reward2 = (1, -1) if agent1_turn else (-1, 1)
                    done = True
                elif not game.board.get_empty_positions():
                    reward1 = reward2 = 0  # 引き分けの場合の報酬
                    done = True
                else:
                    reward1 = reward2 = -0.01  # 通常のターンの少額のペナルティ
                    agent1_turn = not agent1_turn  # ターンを交代

                next_state = game.get_encoded_state()
                next_state = np.reshape(next_state, [1, self.state_size])

                if agent1_turn:
                    agent1.remember(state, action, reward1, next_state, done)
                else:
                    agent2.remember(state, action, reward2, next_state, done)

                state = next_state

                if done:
                    print(f"Episode {e+1}/{num_episodes} - Agent1 Reward: {reward1}, Agent2 Reward: {reward2}")
                    if e % target_update_freq == 0:
                        agent1.update_target_model()
                        agent2.update_target_model()
                    break

            if len(agent1.memory) > batch_size:
                agent1.replay(batch_size)
            if len(agent2.memory) > batch_size:
                agent2.replay(batch_size)





# if __name__ == "__main__":
#     game = QuartoGame()
#     state_size = 4 * 4 * 4
#     action_size = 16
#     agent1 = DQNAgent(state_size=state_size, action_size=action_size)
#     agent2 = DQNAgent(state_size=state_size, action_size=action_size)
#     agent1.train_agents(agent1, agent2, game)

#     # モデルの保存
#     agent1.save_model("agent1_model.pth")
#     agent2.save_model("agent2_model.pth")


