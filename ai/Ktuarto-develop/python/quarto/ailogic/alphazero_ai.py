from .montecarlo_ai import *
from ..gameobject import board, piece, box
from ..NueralNet import AlphaZeroNet, Parameter
import numpy as np
import torch

class AlphaZeroAI(Montecarlo):
    def __init__(self):
        super().__init__()
        self.model = AlphaZeroNet()
        self.param = Parameter()

    def choice(self, in_board, in_box):
        in_board = board.HiTechBoard(in_board)
        in_box = box.Box(in_box)
        
        # ニューラルネットワークを使用して盤面を評価し、手を選択
        board_tensor = torch.tensor(in_board.toNumList()).float().unsqueeze(0).unsqueeze(0)
        value, policy = self.model(board_tensor)
        
        # policyに基づいて手を選択
        policy = policy.detach().numpy().flatten()
        res_piece = in_box.piecelist[np.argmax(policy)]
        
        res_piece = res_piece.toDict()
        res_call = "Quarto" if in_board.isQuarto() else "Non"

        return {'piece': res_piece, 'call': res_call}

    def put(self, in_board, in_piece):
        in_board = board.HiTechBoard(in_board)
        in_piece = piece.Piece.getInstance(in_piece)
        
        # ニューラルネットワークを使用して盤面を評価し、手を選択
        board_tensor = torch.tensor(in_board.toNumList()).float().unsqueeze(0).unsqueeze(0)
        value, policy = self.model(board_tensor)
        
        # policyに基づいて手を選択
        policy = policy.detach().numpy().flatten()
        empty_positions = np.where(in_board.onboard == None)
        best_position = np.argmax(policy[empty_positions])
        res_left, res_top = empty_positions[0][best_position], empty_positions[1][best_position]
        
        in_board.setBoard(res_left, res_top, in_piece)
        res_call = "Quarto" if in_board.isQuarto() else "Non"

        return {'call': res_call, 'left': res_left, 'top': res_top}