import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps






# アプリケーション画面
class App(tk.Frame):
    global dataset
    global model
    global device
    global half
    global save_dir
    global colors
    global names
    def __init__(self, window_name):
        self.root = tk.Tk()           # rootメインウィンドウの設定
        super().__init__(self.root)   # tkinterクラスを継承
        self.root.title(window_name)    # ウィンドウタイトル
        # メインフレームの作成
        self.frame = tk.Frame(self.root)
        self.frame.pack(fill=tk.BOTH, padx=20, pady=10) # 設置
        # キャンバス作成
        self.canvas = tk.Canvas(self.frame)
        self.canvas.pack(expand=True, fill=tk.BOTH)
        # カメラ起動ボタン
        camera_start_btn = tk.Button(self.frame, text="カメラ起動", command=self.set_camera)
        camera_start_btn.pack(fill='x', side='left', padx=10)

        # 学習かいし
        camera_run_btn = tk.Button(self.frame, text="学習開始", command=self.detect_main)
        camera_run_btn.pack(fill='x', side='left', padx=10)
        # カメラ開始ボタン
        camera_run_btn = tk.Button(self.frame, text="カメラ開始", command=self.recog)
        camera_run_btn.pack(fill='x', side='left', padx=10)

        # カメラ終了ボタン
        camera_exit_btn = tk.Button(self.frame, text="カメラ終了", command=self.destroy_camera)
        camera_exit_btn.pack(fill='x', side='left', padx=10)
        print("初期化完了.")

    # カメラ起動
    def set_camera(self):
        self.capture = cv2.VideoCapture(0)
        print("カメラ起動完了.")

    # カメラ停止
    def destroy_camera(self):
        self.after_cancel(self.camera_run_id)
        self.disp_id = None
        self.capture.release()
        print("カメラ終了.")

    # カメラ表示
    def camera(self):
        # 撮影
        ret, frm = self.capture.read()
        cv_image = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB) # BGR→RGB変換
        # NumPyのndarrayからPillowのImageへ変換
        pil_image = Image.fromarray(cv_image)
        # キャンバスのサイズを取得
        self.canvas_width = self.canvas.winfo_width()
        self.canvas_height = self.canvas.winfo_height()
        # 画像のアスペクト比（縦横比）を崩さずに指定したサイズ（キャンバスのサイズ）全体に画像をリサイズする
        pil_image = ImageOps.pad(pil_image, (self.canvas_width, self.canvas_height))
        # PIL.ImageからPhotoImageへ変換する
        self.photo_image = ImageTk.PhotoImage(image=pil_image)
        # 画像の描画
        self.canvas.create_image(
                self.canvas_width / 2,  # 画像表示位置(Canvasの中心)
                self.canvas_height / 2,
                image=self.photo_image  # 表示画像データ
                )
        # 画面の更新を10msecごとに行う
        self.camera_run_id = self.after(10, self.camera)

    @staticmethod
    def detect_img(model,img):
        # 推論の結果
        conf_thres = 0.25
        augment   = False
        iou_thres = 0.45
        classes   = None
        agnostic_nms = False

                # これはよくわからん,モデルがどうゆう出力かなんちゃら
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=augment)[0]
        t2 = time_synchronized()
        # Apply NMS
        # 何かわからんが、物体検知を行った時に値が格納されるようになっている
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)

        return pred
    # インスタンス化せずにメソッドからメソッドを呼び出せる
    @staticmethod
    def detect_detaset(model):
        source    = "0" # ソース
        stride = int(model.stride.max())  # model stride
        img_size  = 320
        imgsz     = img_size
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        # データローダーを設置
        vid_path, vid_writer = None, None
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)

        return dataset
    @staticmethod
    def detect_bbox(det,img,im0,txt_path,gn,colors,names,s):
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()  # detections per class
            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

        # Write results
        # バウンディングボックスの座標を保存するtxtと、描画するとこを
        for *xyxy, conf, cls in reversed(det):
            save_conf = True
            # 座標をtxtに保存している
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
            with open(txt_path + '.txt', 'a') as f:
                f.write(('%g ' * len(line)).rstrip() % line + '\n')

            label = f'{names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

    def detect_main(self):
        global dataset
        global model
        global device
        global half
        global save_dir
        global colors
        global names
        weights   = "../weights/detect_tomato.pt" # 重み
        project   = "runs/detect" # 結果の保存ディレクトリ階層
        save_txt  = True
        name      = "exp"         # 結果の保存ディレクトリ名
        exist_ok  = ""            # existing project/name ok, do not increment
        device    = ""            # cuda device, i.e. 0 or 0,1,2,3 or cpu

        # 保存ディレクトリの作成
        save_dir = Path(increment_path(Path(project) / name, exist_ok=exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # 初期化
        set_logging()
        device = select_device(device)
        half = device.type != 'cpu'  # half precision only supported on CUDA
        print("device",device)

        # モデル読み込み
        model = attempt_load(weights, map_location=device)  # load FP32 model
        dataset = App.detect_detaset(model)

         # name モデルの名前を取得している
        names = model.module.names if hasattr(model, 'module') else model.names
        # 物体検知を行った時のラベルの色を適当に選出
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]


# ,dataset,model,device,half,save_dir,colors,names
    def recog(self):
        global dataset
        global model
        global device
        global half
        global save_dir
        global colors
        global names
        # ret, frm = self.capture.read()
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            # 学習
            print("学習")
            pred = App.detect_img(model,img)
            # 物体を検知したときにpredに値が入る
            print("pred",pred)

            for i, det in enumerate(pred):  # detections per image
                # 内部も認識した時に一回
                print(",pred内部", det)
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                p = Path(p)  # to Path
                # save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt

                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                print("gn",gn)

                if len(det):
                    # バウンディングボックスのtxtを作成
                    App.detect_bbox(det,img,im0,txt_path,gn,colors,names,s)
                # cv2.imshow(str(p),im0)
                # cv2.waitKey(1)
                cv_image = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB) # BGR→RGB変換
                # NumPyのndarrayからPillowのImageへ変換
                pil_image = Image.fromarray(cv_image)
                # キャンバスのサイズを取得
                self.canvas_width = self.canvas.winfo_width()
                self.canvas_height = self.canvas.winfo_height()
                # 画像のアスペクト比（縦横比）を崩さずに指定したサイズ（キャンバスのサイズ）全体に画像をリサイズする
                pil_image = ImageOps.pad(pil_image, (self.canvas_width, self.canvas_height))
                # PIL.ImageからPhotoImageへ変換する
                self.photo_image = ImageTk.PhotoImage(image=pil_image)
                # 画像の描画
                self.canvas.create_image(
                        self.canvas_width / 2,  # 画像表示位置(Canvasの中心)
                        self.canvas_height / 2,
                        image=self.photo_image  # 表示画像データ
                        )
                # cv_image = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB) # BGR→RGB変換
                # NumPyのndarrayからPillowのImageへ変換
            break
        self.camera_run_id = self.after(100, self.recog)






if __name__ == '__main__':
    app = App("application")
    app.mainloop()