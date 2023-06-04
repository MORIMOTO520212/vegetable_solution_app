# Vegetable Solusion Application - ソシューション開発I,II
<img src="https://img.shields.io/badge/-Python3.9.0-3776AB.svg?logo=python&style=plastic">

>Author: Nitanda, Morimoto, Kodama, Itakura, Kudou

## やること
detectクラスを作って、  
```python
def set_camera(self):
    self.model = detect(weightsPath, cameraId) # モデル読み込みとカメラ起動

def camera(self):
    img = model.search()
```

## 説明
野菜の葉状疾患の検出と、収穫可能な実の検出を行うシステム

## 機能
1. モジュール
2. API

## 使い方
```pip install -r requirements.txt```  
```python -V```  
```python app.py```


## 仕様
使用しているYOLOv7  
https://github.com/WongKinYiu/yolov7  

ライセンスなし