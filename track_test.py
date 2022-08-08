import sys
sys.path.insert(0, './yolov5')

from yolov5.utils.google_utils import attempt_download
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, check_imshow, xyxy2xywh,\
    increment_path,calculate_IoU,delimg,compute_IOU,detooth,xywh2xyxy
from yolov5.utils.torch_utils import select_device, time_synchronized
from yolov5.utils.plots import plot_one_box
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy
from model_load import load_model



def compute_color_for_id(label):
    """
    Simple function that adds fixed color depending on the id
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def calculate_U(predicted_bound, ground_truth_bound):
    """
    computing the IoU of two boxes.
    Args:
        box: (xmin, ymin, xmax, ymax),通过左下和右上两个顶点坐标来确定矩形位置
    Return:
        IoU: IoU of box1 and box2.
    """
    pxmin, pymin, pxmax, pymax = predicted_bound
    #print("预测框P的坐标是：({}, {}, {}, {})".format(pxmin, pymin, pxmax, pymax))
    gxmin, gymin, gxmax, gymax = ground_truth_bound
    #print("原标记框G的坐标是：({}, {}, {}, {})".format(gxmin, gymin, gxmax, gymax))

    # 求相交矩形的左下和右上顶点坐标(xmin, ymin, xmax, ymax)
    xmin = min(pxmin, gxmin)  # 得到左下顶点的横坐标
    ymin = min(pymin, gymin)  # 得到左下顶点的纵坐标
    xmax = max(pxmax, gxmax)  # 得到右上顶点的横坐标
    ymax = max(pymax, gymax)  # 得到右上顶点的纵坐标
    return xmin, ymin, xmax, ymax

def getattr(object, name, default=None):  # known special case of getattr
    """
    getattr(object, name[, default]) -> value

    Get a named attribute from an object; getattr(x, 'y') is equivalent to x.y.
    When a default argument is given, it is returned when the attribute doesn't
    exist; without it, an exception is raised in that case.
    """
    pass
def detect(opt, model, model0):
    source, yolo_weights, deep_sort_weights, show_vid, save_vid, save_txt, imgsz, evaluate,nosave,project,name ,exist_ok,yolo_weigths_tooth= \
         opt.source, opt.yolo_weights, opt.deep_sort_weights, opt.show_vid, opt.save_vid, \
            opt.save_txt, opt.img_size, opt.evaluate,opt.nosave,opt.project,opt.name,opt.exist_ok,opt.yolo_weights_tooth
    #nosave= False
    #print(project)
    #print(name)
    im0_sum=[]
    #save_txt=True
    save_img = not nosave and not source.endswith('.txt')

    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')
    # Directories
    # 预测路径是否存在，不存在新建，按照实验文件以此递增新建
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
    # print(cfg.DEEPSORT, cfg.DEEPSORT.REID_CKPT)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
#保存路径-可改
    '''
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder
    '''
    half = device.type != 'cpu'  # half precision only supported on CUDA
    # Load model
    
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    names_tooth = model0.module0.names if hasattr(model0, 'module') else model0.names  # get class names
    if half:
        model.half()  # to FP16
    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays

    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz,stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    #save_path = str(Path(out))
    # extract what is in between the last '/' and last '.'
    #txt_file_name = source.split('/')[-1].split('.')[0]
    #txt_path = str(Path(out)) + '/' + txt_file_name + '.txt'
    #print(txt_path)
    print('dataset=',enumerate(dataset))
    mean_time = 0
    x=0
    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        #print('frame_idx=',frame_idx)

        current_time = cv2.getTickCount()
        img = torch.from_numpy(img).to(device)

        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()
        #print('pre=',pred)
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            #print('i=',i)
            #print('enu=',det)
            if webcam:  # batch_size >= 1
                p, s, im0, frame= path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset,'frame',0)
            im0_sum.append(im0)
            #s += '%gx%g ' % img.shape[2:]  # print string
            #save_path = str(Path(out) / Path(p).name)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            save_path_img=str(save_dir / p.stem)
            (save_dir / p.stem).mkdir(parents=True, exist_ok=True)
            txt_path = str(save_dir / 'labels' / p.stem) +'.txt'  # im.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            #gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            #imc = im0.copy() if save_crop else im0  # for save_crop
            #annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                out_xyxy = det[:, 0:6].cpu().numpy()

                #print('xyxy=',xyxy)

                #print('xywhs=',xywhs.cpu())
                confs = det[:, 4]
                clss = det[:, 5]
                #print('con=',confs.cpu(),clss.cpu())
                # pass detections to deepsort 去虚警
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss, im0)
                #print('outputs=',outputs)

                if len(outputs)>0:
                    out_tooths=load_model(model0,img,path,opt,dataset,im0s,save_dir,frame_idx)
                    output_qc=outputs
                    output_tooth = out_tooths
                    output_final1=[]
                    #output_final2=[]
                    '''
                    for out in outputs:
                        if out[-1]==0:
                            output_qc.append(out)
                        else :
                            output_tooth.append(out)
                    '''
                    #print('output_qc=',output_qc)
                    for qc in output_qc:
                        for tooth in output_tooth:
                            #print('iou=',compute_IOU(qc[0:4],tooth[0:4]))
                            if(compute_IOU(qc[0:4],tooth[0:4])>0.5 or calculate_IoU(qc[0:4],tooth[0:4])>0.5):
                                #print('qc=',qc)
                                qc = numpy.append(qc, tooth[4])
                                #qc.append(tooth[5])
                                output_final1.append(qc)
                                #output_final2.append(tooth)
                                break
                    '''            
                    for i in range(len(output_final1)):
                        j=i+1
                        qc_i=output_final1[i]
                        for j in range(len(output_final1)):
                            qc_j=output_final1[j]
                            if(compute_IOU(qc_i[0:4],qc_j[0:4])>0):
                                qc=[]
                                xmin = min(qc_i[0], qc_j[0])  # 得到左上顶点的横坐标
                                ymin = min(qc_i[1], qc_j[1])  # 得到左上顶点的纵坐标
                                xmax = min(qc_i[2], qc_j[2])  # 得到右下顶点的横坐标
                                ymax = min(qc_i[3], qc_j[3])  # 得到右下顶点的纵坐标
                                qc.append(xmin)
                                qc.append(ymin)
                                qc.append(xmax)
                                qc.append(ymax)
                                qc.append(qc_i[4])
                                qc.append(qc_i[5])
                                print('qc=',qc)
                                output_final2.append(qc)
                            else:
                                #output_final2.append(qc_j)
                                if qc_i not in  output_final2:
                                    output_final2.append(qc_i)
                    '''
                    outputs = output_final1
                    #outputs2 = output_final2
                    #print('outputs=', outputs)
                if len(outputs) > 0:
                    ou = []
                    for out1 in outputs:
                        print(out1, type(out1))
                        o = out1.tolist()
                        out1 = o
                        print(out1, type(out1))
                        for out2 in out_xyxy:
                            if compute_IOU(out1[0:4], out2[0:4]) > 0.8:
                                print(out2[4])
                                out1.append(out2[4])
                                ou.append(out1)
                                break
                    print('ou=', ou)
                    outputs = ou

                if len(outputs) > 0:
                    for j, (output) in enumerate(outputs):
                        bboxes = output[0:4]
                        #print('bboxes=',bboxes)
                        id = output[4]
                        cls = output[5]
                        conf = output[7]
                        #print('output=',output)
                        #label = f'{id} {names[c]} {conf:.2f}'
                        #print('label=',label)
                        c = int(cls)
                        #print(names[c])
                        #color = compute_color_for_id(id)
                        #plot_one_box(bboxes, im0, label=label, color=color, line_thickness=2)

                        if save_txt:
                            # to MOT format
                            bbox_top = output[0]
                            bbox_left = output[1]
                            bbox_w = output[2]
                            bbox_h = output[3]
                            # Write MOT compliant results to file
                            with open(txt_path,'a') as f:
                                if names[c]=='qc':
                                    f.write(('%g ' * 2 + '%s ' + '%g ' * 4 + '%g ' +'\n') % (frame_idx, id, names_tooth[int(output[6])], bbox_top,
                                                                                  bbox_left, bbox_w, bbox_h, conf))  # label format
                                    print(('%g ' * 2 + '%s ' + '%g ' * 4 + '%g ' +'\n') % (frame_idx, id, names_tooth[int(output[6])], bbox_top,
                                                                                  bbox_left, bbox_w, bbox_h, conf))

            else:
                deepsort.increment_ages()

            # Print time (inference + NMS)
            #cv2.imwrite(save_path, im0)
            #print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            '''if show_vid:
                current_time = (cv2.getTickCount() - current_time) / cv2.getTickFrequency()
                if mean_time == 0:
                    mean_time = current_time
                else:
                    mean_time = mean_time * 0.95 + current_time * 0.05

                # videowriter = cv2.VideoWriter("b.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (h, w))
                # videowriter.write(output)

                cv2.putText(im0, 'FPS: {}'.format(int(1 / mean_time * 10) / 10),
                            (40, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
                #cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration'''

            # Save results (image with detections)
#no save_img
            #frame=delimg(txt_path)

            if save_img:
                if dataset.mode=='image':
                    cv2.imwrite(save_path,im0)
                else:
                    '''
                    print('x=', x)
                    #if str(x) in frame:
                    save_image_dir = os.path.join(save_path_img, '%s.jpg' % x)
                    #x=x+1
                    print('save_image_dir: ', save_image_dir)
                    cv2.imwrite(save_image_dir, im0)
                    x = x + 1
                    '''
                    if save_vid:
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'


                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer.write(im0)
    if save_txt or save_vid:
        #print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)
    print('Done. (%.3fs)' % (time.time() - t0))
    return txt_path, save_path_img, im0_sum,save_dir

def yolo_predict(fpath,model,model0):
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_weights', type=str, default='weights/best.pt', help='model.pt path')
    parser.add_argument('--yolo_weights_tooth', type=str, default='weights/tooth_best.pt', help='model.pt path')
    parser.add_argument('--deep_sort_weights', type=str, default='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7', help='ckpt.t7 path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default=fpath, help='source')
    #parser.add_argument('--output', type=str, default='inference/output/', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', default=True, action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', default=True, action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', default=True, action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--project', default='inference/output', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    args, unknown = parser.parse_known_args()
    args.img_size = check_img_size(args.img_size)

    with torch.no_grad():
        path=detect(args, model, model0)

        txt_path=path[0]
        img_path=path[1]
        im0_sum=path[2]
        save_dir = path[3]
        print(txt_path,img_path)
        if os.path.exists(txt_path) == False :
            return img_path
        frame = delimg(txt_path)
        '''
        for root, dirs, files in os.walk(img_path):
            for name in files:
                if name.split('.')[0] not in frame:  # 填写规则
                    os.remove(os.path.join(root, name))
        '''
        f = open(txt_path)
        lines = f.readlines()
        boxs = []
        idex = -1
        max_conf = 0.0
        for line in lines:
            outputs = line.split(' ')
            output = outputs[:-1]  # 换行符
            if output[0] in frame:
                if output[0] == idex:
                    id, name, box, conf = output[1], output[2], output[3:7], output[7]
                    boxs_del = []  # 与box重合要删除的框
                    for i in boxs:
                        if calculate_IoU(list(map(int, i)), list(map(int, box))) > 0:
                            box = calculate_U(list(map(int, i)), list(map(int, box)))
                            boxs_del.append(i)
                    boxs.append(box)
                    max_conf = max(float(conf), max_conf)
                    for i in boxs_del:
                        boxs.remove(i)
                else:
                    if idex != -1:
                        im0 = im0_sum[int(idex)]
                        for box in boxs:
                            id,name = output[1],output[2]
                            bboxes = list(map(int, box))
                            #label = f'{name}{max_conf:.2f}'
                            label = f'{name}'
                            #color = compute_color_for_id(int(id))
                            #im0=cv2.imread(os.path.join(img_path,im))
                            print('label', label)
                            #if (max_conf >= 0.6):
                            #    color = (0, 0, 210)
                            #else:
                            #    color = (0, 210, 210)'''
                            color = (0, 0, 210)
                            plot_one_box(bboxes, im0, label=label, color=color, line_thickness=2)
                            save_image_dir = os.path.join(img_path, '%s.jpg' % idex)
                            # x=x+1
                            print('save_image_dir: ', save_image_dir)
                            cv2.imwrite(save_image_dir, im0)
                        boxs.clear()
                        boxs.append(output[3:7])
                        idex = output[0]
                        max_conf = float(output[7])
                    else:
                        boxs.append(output[3:7])
                        idex = output[0]
                        max_conf = float(output[7])
    return img_path


