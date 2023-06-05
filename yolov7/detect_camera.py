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


def detect():
    # webcam    = True
    source    = "0" # ソース
    weights   = "../weights/detect_tomato.pt" # 重み
    img_size  = 320 # inference size (pixels)
    # save_img  = False # save inference images
    view_img  = False  # display results
    conf_thres = 0.25  # object confidence threshold
    name      = "exp"         # 結果の保存ディレクトリ名
    project   = "runs/detect" # 結果の保存ディレクトリ階層
    save_txt  = True
    save_conf = True
    imgsz     = img_size
    iou_thres = 0.45          # IOU threshold for NMS
    classes   = None
    trace     = False
    device    = ""            # cuda device, i.e. 0 or 0,1,2,3 or cpu
    exist_ok  = ""            # existing project/name ok, do not increment
    augment   = False         # augmented inference
    trace     = False         # don`t trace model
    agnostic_nms = False      # class-agnostic NMS

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
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    print("imgsz",imgsz)

    # if trace:
    #     model = TracedModel(model, device, img_size)
    #     print("imgsz43")

    # if half:
    #     model.half()  # to FP16
    #     print("imgsz4365")

    # 第2分類器ResNetの読み込み
    # classify = False
    # if classify:
    #     modelc = load_classifier(name='resnet101', n=2)  # initialize
    #     modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # データローダーを設置
    vid_path, vid_writer = None, None
    view_img = check_imshow()
    cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams(source, img_size=imgsz, stride=stride)


    # Get names and colors
    # name モデルの名前を取得している
    names = model.module.names if hasattr(model, 'module') else model.names
    # 物体検知を行った時のラベルの色を適当に選出
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    print("gfdagfgdsgfdsfdgssgfsfgfsgdf")
    
    # これ以降がずっと繰り返し
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        print("img",img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        t1 = time_synchronized()
        # これはよくわからん,モデルがどうゆう出力かなんちゃら
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=augment)[0]
        t2 = time_synchronized()
        # Apply NMS
        # 何かわからんが、物体検知を行った時に値が格納されるようになっている
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        t3 = time_synchronized()

        # ９６−１００行くらいで学習
       
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count

            p = Path(p)  # to Path
            # save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt

            # gnは画面の大きさが格納
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            print("gn",gn)
            # 物体を検知したらdetに値が格納される
            if len(det):
                # Rescale boxes from img_size to im0 size
                # 画像の大きさとかかな？
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                print('det',det)

                # Print results
                # 分類する種類の名前を格納
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                # バウンディングボックスの座標を保存するtxtと、描画するとこを
                for *xyxy, conf, cls in reversed(det):
                    # 座標をtxtに保存している
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

<<<<<<< Updated upstream

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

=======
>>>>>>> Stashed changes
    print(f'Done. ({time.time() - t0:.3f}s)')

if __name__ == '__main__':
    detect()