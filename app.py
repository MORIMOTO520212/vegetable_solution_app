import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps
import cv2

class App(tk.Frame):
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
        # カメラ開始ボタン
        camera_run_btn = tk.Button(self.frame, text="カメラ開始", command=self.camera)
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


if __name__ == '__main__':
    app = App("application")
    app.mainloop()