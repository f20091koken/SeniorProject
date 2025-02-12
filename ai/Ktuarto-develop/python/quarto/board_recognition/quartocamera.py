import cv2
import numpy as np
import os
import math
import json
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

class QuartoRecognition:
    def __init__(self):
        """各属性のモデルを読み込む"""
        model_dir = "C:\\Users\\kairi\\Documents\\GitHub\\SeniorProject\\ai\\Ktuarto-develop\\python\\quarto\\board_recognition"
        self.models = {
            'color': load_model(os.path.join(model_dir, 'best_model_color.keras')),
            'shape': load_model(os.path.join(model_dir, 'best_model_shape.keras')),
            'height': load_model(os.path.join(model_dir, 'best_model_height.keras')),
            'hole': load_model(os.path.join(model_dir, 'best_model_hole.keras'))
        }

    def load_and_prepare_image(self, image_path, target_size=(140, 140)):
        """画像を読み込んでモデル用に前処理"""
        img = cv2.imread(image_path)
        img = cv2.resize(img, target_size)
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0
        return img

    def recognize_piece(self, image_path):
        """1つのマスの画像に対して駒の有無を確認し、存在する場合は4つの属性を判定"""
        # 画像を読み込む
        img = cv2.imread(image_path)
        if img is None:
            print(f"画像を読み込めませんでした: {image_path}")
            return "----"
        
        # 駒の有無をチェック
        if check_piece_existence(img):
            return "----"
        
        # 駒が存在する場合は属性を判定
        prepared_img = self.load_and_prepare_image(image_path)
        results = []
        
        # 各属性モデルで判定
        for model_name, model in self.models.items():
            prediction = model.predict(prepared_img, verbose=0)
            predicted_class = int(prediction[0][0] > 0.5)
            results.append(str(predicted_class))
        
        return ''.join(results)

class QuartoCamera:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.recognition = QuartoRecognition()
        self.base_save_dir = "captured_images"
        os.makedirs(self.base_save_dir, exist_ok=True)

    def detect_piece_addition(self):
        """
        カメラを起動し、Sキーが押されたときに盤面の変更を検出して
        新しく駒が置かれた位置を返す
        
        Returns:
            str: "row col" - 新しく駒が置かれた位置（半角スペース区切り）
            None: 駒の追加が検出されなかった場合
        """
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                overlay = create_overlay(frame.shape)
                frame_overlaid = cv2.add(frame, overlay)
                
                cv2.namedWindow('Camera with Overlay', cv2.WINDOW_NORMAL)
                cv2.imshow('Camera with Overlay', cv2.resize(frame_overlaid, (960, 540)))
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    return None
                elif key == ord('s'):
                    # 画像を保存して処理
                    original_path = self._save_image(frame, True)
                    self._save_image(frame_overlaid, False)
                    
                    # 画像を処理して新しい駒の位置を取得
                    position = self._process_image(original_path)
                    if position is not None:
                        return position
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def _save_image(self, image, is_original):
        """画像を保存し、保存したファイルパスを返す"""
        overlay_name = "square"
        image_type = "original" if is_original else "overlaid"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(self.base_save_dir, timestamp, overlay_name)
        os.makedirs(save_dir, exist_ok=True)
        
        filename = f"{image_type}_{overlay_name}.jpg"
        full_path = os.path.join(save_dir, filename)
        
        cv2.imwrite(full_path, image)
        return full_path

    def _process_image(self, filename):
        """
        画像を処理して新しく追加された駒の位置を返す
        
        Returns:
            str: "row col" - 新しく駒が置かれた位置（半角スペース区切り）
            None: 新しい駒が検出されなかった場合
        """
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        if image is None:
            return None

        directory_name = os.path.join(os.path.dirname(filename), "transformed")
        os.makedirs(directory_name, exist_ok=True)

        squares = findSquares(image)
        largest_square = getLargestSquare(squares)

        if largest_square is not None:
            warped = perspectiveTransform(image, largest_square)
            resized = cv2.resize(warped, (800, 800), interpolation=cv2.INTER_AREA)
            brightened = increase_brightness(resized)

            transformed_filename = os.path.join(directory_name, "transformed_800x800.jpg")
            cv2.imwrite(transformed_filename, brightened)

            cropped_dir = os.path.join(directory_name, "cropped")
            os.makedirs(cropped_dir, exist_ok=True)

            # 切り取り座標（既存のまま）
            crop_coordinates = [
                (325, 25, 465, 165),   # 位置 0
                (425, 125, 565, 265),  # 位置 1
                (525, 225, 665, 365),  # 位置 2
                (625, 325, 765, 465),  # 位置 3
                (225, 125, 365, 265),  # 位置 4
                (325, 225, 465, 365),  # 位置 5
                (425, 325, 565, 465),  # 位置 6
                (525, 425, 665, 565),  # 位置 7
                (125, 225, 265, 365),  # 位置 8
                (225, 325, 365, 465),  # 位置 9
                (325, 425, 465, 565),  # 位置 10
                (425, 525, 565, 665),  # 位置 11
                (25, 325, 165, 465),   # 位置 12
                (125, 425, 265, 565),  # 位置 13
                (225, 525, 365, 665),  # 位置 14
                (325, 625, 465, 765)   # 位置 15
            ]

            new_board_state = [[None for _ in range(4)] for _ in range(4)]
            
            # 各マスの処理
            for i, (left, upper, right, lower) in enumerate(crop_coordinates):
                cropped = brightened[upper:lower, left:right]
                cropped_filename = os.path.join(cropped_dir, f"cropped_{i:02d}.jpg")
                cv2.imwrite(cropped_filename, cv2.resize(cropped, (140, 140)))
                
                piece_features = self.recognition.recognize_piece(cropped_filename)
                
                row = i // 4
                col = i % 4
                new_board_state[row][col] = piece_features

            # 盤面情報を保存し、前回の状態を取得
            previous_state = save_board_state(new_board_state)
            
            # 新しく追加された駒の位置を検出し、文字列として返す
            if previous_state is not None:
                for i in range(4):
                    for j in range(4):
                        if (previous_state[i][j] == "----" or 
                            previous_state[i][j] is None) and new_board_state[i][j] != "----":
                            return f"{i} {j}"  # 半角スペースで区切った文字列として返す

        return None

def save_board_state(current_state, filename="board_state.json"):
    """盤面情報をJSONファイルに保存"""
    try:
        # 既存のデータを読み込む
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                previous_state = data.get('current_state')
        except (FileNotFoundError, json.JSONDecodeError):
            previous_state = None

        # 新しいデータを保存
        state_data = {
            'timestamp': datetime.now().isoformat(),
            'current_state': current_state,
            'previous_state': previous_state
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, indent=2, ensure_ascii=False)
        return previous_state
    except Exception as e:
        print(f"盤面の保存中にエラーが発生しました: {e}")
        return None

def print_board_state(board_state):
    """盤面の状態を整形して表示"""
    print("\n=== 現在の盤面状態 ===")
    print("    0      1      2      3   ")
    print("  -------------------------")
    for i, row in enumerate(board_state):
        print(f"{i}|", end=" ")
        for piece in row:
            if piece is None:
                print("----", end=" | ")
            else:
                print(piece, end=" | ")
        print("\n  -------------------------")

def compare_states(current_state, previous_state):
    """現在の盤面と前回の盤面を比較し、変更があった箇所を表示"""
    if previous_state is None:
        print("\n前回の盤面情報がありません")
        return
    
    print("\n=== 盤面の変更点 ===")
    for i in range(4):
        for j in range(4):
            if current_state[i][j] != previous_state[i][j]:
                old_piece = "----" if previous_state[i][j] is None else previous_state[i][j]
                new_piece = "----" if current_state[i][j] is None else current_state[i][j]
                print(f"位置 ({i},{j}): {old_piece} → {new_piece}")

def check_piece_existence(image):
    """
    グレースケール画像を分析して駒の有無を判定する
    
    Parameters:
    image (numpy.ndarray): 分析する画像（カラー）
    
    Returns:
    bool: False if piece exists, True if no piece
    """
    # グレースケールに変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 125より大きい値のピクセル数をカウント
    white_pixels = np.sum(gray > 125)
    
    # 5000ピクセル以上であれば駒なしと判定
    return white_pixels >= 3000

def create_overlay(shape):
    overlay = np.zeros(shape, dtype=np.uint8)
    height, width = shape[:2]
    
    # 正方形オーバーレイ
    size = int(min(width, height) * 3.5 // 10)
    top_left = ((width - size) // 2, (height - size) // 2)
    bottom_right = (top_left[0] + size, top_left[1] + size)
    color = (0, 255, 0)  # 緑色
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
    
    return cv2.addWeighted(overlay, 0.3, np.zeros_like(overlay), 0, 0)

def save_image(image, base_dir, is_original):
    """画像を保存し、保存したファイルパスを返す"""
    overlay_name = "square"
    image_type = "original" if is_original else "overlaid"
    
    # タイムスタンプベースのサブディレクトリを作成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(base_dir, timestamp, overlay_name)
    os.makedirs(save_dir, exist_ok=True)
    
    filename = f"{image_type}_{overlay_name}.jpg"
    full_path = os.path.join(save_dir, filename)
    
    cv2.imwrite(full_path, image)
    print(f"{image_type.capitalize()} 画像を保存しました: {full_path}")
    return full_path

def angle(pt1, pt2, pt0) -> float:
    dx1 = float(pt1[0,0] - pt0[0,0])
    dy1 = float(pt1[0,1] - pt0[0,1])
    dx2 = float(pt2[0,0] - pt0[0,0])
    dy2 = float(pt2[0,1] - pt0[0,1])
    v = math.sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2))
    return (dx1*dx2 + dy1*dy2) / v

def findSquares(image, areaThreshold=1000, aspect_ratio_threshold=0.1):
    squares = []
    gray0 = np.zeros(image.shape[:2], dtype=np.uint8)

    rows, cols, _channels = map(int, image.shape)
    pyr = cv2.pyrDown(image, dstsize=(cols//2, rows//2))
    timg = cv2.pyrUp(pyr, dstsize=(cols, rows))

    for c in range(0, 3):
        cv2.mixChannels([timg], [gray0], (c, 0))

        for l in range(0, 11):
            if l == 0:
                gray = cv2.Canny(gray0, 50, 5)
                gray = cv2.dilate(gray, None)
            else:
                gray = cv2.threshold(gray0, (l+1)*255/11, 255, cv2.THRESH_BINARY)[1]

            contours, _ = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                arclen = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, arclen*0.02, True)
                area = abs(cv2.contourArea(approx))

                if approx.shape[0] == 4 and area > areaThreshold and cv2.isContourConvex(approx):
                    maxCosine = max(abs(angle(approx[j%4], approx[j-2], approx[j-1])) for j in range(2, 5))
                    if maxCosine < 0.3:
                        x, y, w, h = cv2.boundingRect(approx)
                        aspect_ratio = abs(1 - float(w) / h)
                        if aspect_ratio < aspect_ratio_threshold:
                            squares.append((approx, area))

    return squares

def getLargestSquare(squares):
    if not squares:
        return None
    return max(squares, key=lambda x: x[1])[0]

def orderPoints(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def perspectiveTransform(image, square):
    pts = orderPoints(square.reshape(4, 2))
    (tl, tr, br, bl) = pts

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def increase_brightness(img, value=50):
    return img  # 現在は何も処理を行っていない

# 使用例
def main():
    camera = QuartoCamera(camera_index=0)
    position = camera.detect_piece_addition()
    if position is not None:
        print(f"新しい駒が検出されました: {position}")  # "1 1" のような形式で出力
    else:
        print("新しい駒は検出されませんでした")

if __name__ == "__main__":
    main()