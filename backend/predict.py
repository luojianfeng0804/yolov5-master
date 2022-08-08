# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --weights yolov5s.pt --source 0  # webcam
                                                             img.jpg  # image
                                                             vid.mp4  # video
                                                             path/  # directory
                                                             path/*.jpg  # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
"""

from effNet import effNet_img
import argparse
import os
import sys
from pathlib import Path
import shutil
import cv2
import torch
import torch.backends.cudnn as cudnn
from PIL.Image import Image
from numpy import random
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from general import imagededep, save_img
from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,plot_one_box,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from backend.flask_id2name import names
import base64

@torch.no_grad()
def predict(opt, model, dataset, fname):
    out, source, view_img, save_img, save_txt, imgsz = \
        opt['output'], opt['source'], opt['view_img'], opt['save_img'], opt['save_txt'], opt['imgsz']
    imgsz *= 2 if len(imgsz) == 1 else 1  # expand
    # Initialize
    out = out + '/' + fname
    device = select_device(opt['device'])  # 选择设备
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA
    # Load model
    #device = select_device(device)
    #model = DetectMultiBackend(weights, device=device, dnn=False)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= (pt or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Dataloader
    #dataset = LoadImages(opt['source'], img_size=imgsz)

    # Get names and colors
    #names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    det_dick={}
    boxes_detected = []  # 检测结果
    imgdata = [] #检测出有框图像
    model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    num=0
    for path, im, im0s, vid_cap, s in dataset:
        torch.cuda.empty_cache()
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference

        pred = model(im, augment=opt['augment'])
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, opt['conf_thres'], opt['iou_thres'], classes=opt['classes'], agnostic=opt['agnostic_nms'],max_det=opt['max_det'])
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            p, s, im0 = path, '', im0s
            save_path = str(Path(out) / Path(p).name)  # 保存路径
            #print(str(Path(out)))
            #print(save_path)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            #print(txt_path)
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            annotator = Annotator(im0, line_width=3, example=str(names))
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string
                #print("det:",type(det))
                #print(det)
                #det_dick[num]=det
                #print(det_dick)
                #view_save_path = str(Path(out)) + '/' + str(num) + '.jpg'
                #cv2.imwrite(view_save_path, im0)
                #num += 1
                # Write results             
                for *xyxy, conf, cls in reversed(det):
                    #print('xyxy:',type(xyxy))
                    #print('conf:', type(conf))
                    #print('cls:',type(cls))
                    #print(xyxy)
                    #print(conf)
                    #print(cls)
                    xyxy_list = (torch.tensor(xyxy).view(1, 4)).view(-1).tolist()
                    #label_str = '%s %.2f' % (names[int(cls)], conf)
                    #label_num=str(num)
                    xyxy_str=str(xyxy)
                    cls_str=str(cls)
                    #label_str.append([xyxy_str,label,cls_str])
                    boxes_detected.append({"name": names[int(cls.item())],
                                           "conf": str(conf.item()),
                                           "bbox": [int(xyxy_list[0]), int(xyxy_list[1]), int(xyxy_list[2]),
                                                    int(xyxy_list[3])]
                                           })
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        #with open(txt_path + '.txt', 'a') as f:
                            #f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format


                    if conf>0.3 :  # Add bbox to image
                        label = names[int(cls)]
                        plot_one_box(xyxy, im0, label=label, color=colors(int(cls), True), line_thickness=3)
                        #im0 = cv2.putText(im0, fname, (50, 300), font, 1.2, (255, 255, 255), 2)
                        img_stream = 'http://39.107.203.111/'+save_path
                        #print('#################################################',img_stream)
                        '''ret, jpeg = cv2.imencode('.jpg', im0)
                        jpeg.tobytes()
                        img_stream = base64.b64encode(jpeg).decode()'''
                        if len(imgdata)==0:
                            imgdata.append(img_stream)
                        if img_stream != imgdata[-1]:
                            imgdata.append(img_stream)
                    cv2.imwrite(save_path,im0)
            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))
            # Save results (image with detections)
            #if save_img:
                #if dataset.mode == 'images':
                #cv2.imwrite(save_path, im0)
    '''            
    path=str(Path(out))
    print(path)
    imagededep(path)
           
    for fileimgs in os.listdir(path):
        for img in fileimgs:
            print(img)
            img_path = path + '/' + img
            img_num=int(Path(img_path).stem)
            image= Image.open(img_path)
            plot_one_box(label_str[img_num][0], image, label=label_str[img_num][1], color=colors(int(label_str[img_num][2]), True), line_thickness=3)
    '''
    '''results = {"results": boxes_detected}
    print(results)
    print(num)'''

    return imgdata


