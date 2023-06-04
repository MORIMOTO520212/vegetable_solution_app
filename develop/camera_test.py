import cv2

# カメラをオープンする
capture = cv2.VideoCapture(0)

# カメラがオープン出来たか？
camera_opened = capture.isOpened()

while camera_opened:

    # フレーム画像の取得
    ret, frame = capture.read()
    
    # 画像の表示
    cv2.imshow("Image", frame)

    if cv2.waitKey(1) != -1:
        # キー入力で終了
        break

capture.release()
cv2.destroyAllWindows()