import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps  # 画像データ用

import cv2


def hello():
    print("hello")

# rootメインウィンドウの設定
root = tk.Tk()
root.title("application") # ウィンドウタイトル
root.geometry("400x300")

# メインフレームの作成
frame = tk.Frame(root)
frame.pack(fill=tk.BOTH, padx=20, pady=10) # 設置

# ボタンの作成
button = tk.Button(frame, text="hello", command=hello)
button.pack() # 設置

root.mainloop()
