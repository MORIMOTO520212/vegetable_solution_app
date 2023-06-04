import cv2

def check_info(last_index):
    for i in range(last_index):
        try:
            cap = cv2.VideoCapture(i)
            if cap is None or not cap.isOpened():
                raise ConnectionError
            print(f"-*- DEVICE_ID: {i} -*-")
            fps = cap.get(cv2.CAP_PROP_FPS)
            contrast = cap.get(cv2.CAP_PROP_CONTRAST)
            saturation = cap.get(cv2.CAP_PROP_SATURATION)
            gamma = cap.get(cv2.CAP_PROP_GAMMA)
            print(f"FPS: {fps}")
            print(f"Contrast: {contrast}")
            print(f"Saturation: {saturation}")
            print(f"gamma: {gamma}")
        except ConnectionError:
           pass
