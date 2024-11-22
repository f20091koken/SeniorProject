class Piece:
    def __init__(self, shape, color, height, top):
        self.shape = shape  # 形状（例：丸、四角）
        self.color = color  # 色（例：黒、白）
        self.height = height  # 高さ（例：高い、低い）
        self.top = top  # トップ（例：穴あり、穴なし）

    def __repr__(self):
        return f"Piece(shape={self.shape}, color={self.color}, height={self.height}, top={self.top})"

# 駒の特性を定義
pieces = [
    Piece(shape='round', color='black', height='tall', top='solid'),
    Piece(shape='round', color='black', height='tall', top='hollow'),
    Piece(shape='round', color='black', height='short', top='solid'),
    Piece(shape='round', color='black', height='short', top='hollow'),
    Piece(shape='round', color='white', height='tall', top='solid'),
    Piece(shape='round', color='white', height='tall', top='hollow'),
    Piece(shape='round', color='white', height='short', top='solid'),
    Piece(shape='round', color='white', height='short', top='hollow'),
    Piece(shape='square', color='black', height='tall', top='solid'),
    Piece(shape='square', color='black', height='tall', top='hollow'),
    Piece(shape='square', color='black', height='short', top='solid'),
    Piece(shape='square', color='black', height='short', top='hollow'),
    Piece(shape='square', color='white', height='tall', top='solid'),
    Piece(shape='square', color='white', height='tall', top='hollow'),
    Piece(shape='square', color='white', height='short', top='solid'),
    Piece(shape='square', color='white', height='short', top='hollow'),
]