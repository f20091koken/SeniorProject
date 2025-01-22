import cv2
import numpy as np
import os
from datetime import datetime
import math
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# 前のコードの関数は同じなので省略（create_overlay から perspectiveTransform まで）
def create_overlay(shape):
    overlay = np.zeros(shape, dtype=np.uint8)
    height, width = shape[:2]
    
    # 正方形オーバーレイ
    size = min(width, height) // 2
    top_left = ((width - size) // 2, (height - size) // 2)
    bottom_right = (top_left[0] + size, top_left[1] + size)
    color = (0, 255, 0)  # 緑色
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
    
    return cv2.addWeighted(overlay, 0.3, np.zeros_like(overlay), 0, 0)

def save_image(image, base_dir, is_original):
    overlay_name = "square"
    image_type = "original" if is_original else "overlaid"
    
    save_dir = os.path.join(base_dir, overlay_name)
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{image_type}_{overlay_name}_{timestamp}.jpg"
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
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def load_and_prepare_image(image_path, target_size=(140, 140)):
    # 画像を読み込んでモデル用に前処理
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)  # 140x140にリサイズ
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # 正規化
    return img

def process_image(filename):
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    if image is None:
        print("Image file open error -", filename)
        return

    base_name = os.path.splitext(os.path.basename(filename))[0]
    directory_name = f"{base_name}_transformed"
    os.makedirs(directory_name, exist_ok=True)

    squares = findSquares(image)
    largest_square = getLargestSquare(squares)

    if largest_square is not None:
        warped = perspectiveTransform(image, largest_square)
        resized = cv2.resize(warped, (800, 800), interpolation=cv2.INTER_AREA)
        brightened = increase_brightness(resized, value=30)

        transformed_filename = f"{directory_name}/{base_name}_transformed_800x800.jpg"
        cv2.imwrite(transformed_filename, brightened)
        print(f"Transformed, brightened, and resized square has been saved as '{transformed_filename}'")

        # 正方形オーバーレイの場合の切り取り座標
        crop_coordinates = [
            (325, 25, 465, 165),
            (425, 125, 565, 265),
            (525, 225, 665, 365),
            (625, 325, 765, 465),
            (225, 125, 365, 265),
            (325, 225, 465, 365),
            (425, 325, 565, 465),
            (525, 425, 665, 565),
            (125, 225, 265, 365),
            (225, 325, 365, 465),
            (325, 425, 465, 565),
            (425, 525, 565, 665),
            (25, 325, 165, 465),
            (125, 425, 265, 565),
            (225, 525, 365, 665),
            (325, 625, 465, 765)
        ]

        cropped_dir = os.path.join(directory_name, "cropped")
        os.makedirs(cropped_dir, exist_ok=True)

        # モデルの読み込み
        model = load_model('piece_classifier_model_fold_3.h5')

        print("\n=== 画像認識結果 ===")
        print("位置: 認識結果")
        print("-----------------")

        for i, (left, upper, right, lower) in enumerate(crop_coordinates, 1):
            cropped = brightened[upper:lower, left:right]
            
            # 画像認識のための前処理
            cropped_for_recognition = cv2.resize(cropped, (140, 140), interpolation=cv2.INTER_AREA)
            
            # 保存用の画像も同じサイズに
            cropped_filename = f"{cropped_dir}/{base_name}_cropped_{i:02d}.jpg"
            cv2.imwrite(cropped_filename, cropped_for_recognition)

            # 画像認識の実行
            prepared_img = load_and_prepare_image(cropped_filename)
            prediction = model.predict(prepared_img, verbose=0)  # verboseを0に設定して余分な出力を抑制
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class]
            
            # 結果の出力
            print(f"位置 {i:2d}: クラス {predicted_class} (確信度: {confidence:.2f})")

        print("=================")

    else:
        print("No squares detected")

def main():
    camera_index = 0  # カメラのインデックスを適切に設定してください
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    base_save_dir = "captured_images"
    os.makedirs(base_save_dir, exist_ok=True)

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
            break
        elif key == ord('s'):
            original_path = save_image(frame, base_save_dir, True)
            save_image(frame_overlaid, base_save_dir, False)
            process_image(original_path)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()