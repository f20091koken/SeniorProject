# # graphics.py

# import cv2
# import numpy as np

# def process_board_image(image_path):
#     # 画像を読み込む
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     if img is None:
#         raise ValueError("Image not found or cannot be opened.")
    
#     # ここで画像処理を行い、盤面の状態を解析する
#     # 例えば、ピースの位置を認識する処理など
#     # この例では、単純な処理で空の4x4の盤面を返します
#     board_state = [[None for _ in range(4)] for _ in range(4)]
    
#     # 解析した結果を基に、board_stateを更新する処理を追加する
#     # 例えば、以下のように仮のデータを設定します
#     # board_state[0][0] = '0001'  # 例として一つのピースを配置
    
#     return board_state

#aaaaa