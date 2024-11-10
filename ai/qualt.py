import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from QuartoSystem import Piece, Board, QuartoGame

# DQN用のニューラルネットワークの定義
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# DQNエージェントの定義
class QuartoAI:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # 割引率
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32

        # DQNモデルと最適化アルゴリズムの初期化
        self.model = DQN(state_size, action_size).to('cpu')
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # ε-greedy法に基づいた行動選択
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).to('cpu')
        with torch.no_grad():
            action_values = self.model(state)
        return torch.argmax(action_values).item()

    def replay(self):
    # メモリのサイズがバッチサイズ以下の場合は処理を行わない
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).to('cpu')
            next_state = torch.FloatTensor(next_state).to('cpu')
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model(next_state)).item()
            target_f = self.model(state)
            target_f[0, action] = target  # 正しくインデックスを設定
            # ネットワークの更新
            output = self.model(state)
            loss = self.loss_fn(output, target_f)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def save(self, name):
        torch.save(self.model.state_dict(), name)


# クアルトゲームのシミュレーションと学習の実行
if __name__ == "__main__":
    state_size = 16  # 状態ベクトルのサイズ (4x4 のボード)
    action_size = 16  # 置く場所の数
    agent = QuartoAI(state_size, action_size)
    episodes = 1000
    for e in range(episodes):
        game = QuartoGame()
        state = np.reshape(game.board.get_state(), [1, state_size])
        for time in range(500):
            action = agent.act(state)
            row, col = divmod(action, 4)  # 行と列を求める
            piece = game.select_piece_for_opponent()
            game.board.place_piece(row, col, piece)
            reward = 1 if game.board.check_win() else 0
            done = reward == 1 or game.check_draw()
            next_state = np.reshape(game.board.get_state(), [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"episode: {e}/{episodes}, score: {time}, epsilon: {agent.epsilon:.2}")
                break
        agent.replay()
    agent.save("quarto_ai.pth")
