from yolov5.models.experimental import attempt_load
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, check_imshow, xyxy2xywh,\
    increment_path,calculate_IoU,delimg,compute_IOU,detooth
from pathlib import Path
import numpy
from yolov5.utils.plots import plot_one_box
from yolov5.utils.torch_utils import select_device, time_synchronized
import cv2
import os
def compute_color_for_id(label):
    """
    Simple function that adds fixed color depending on the id
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)
def load_model(model,img ,path,opt,dataset,im0s,save_dir,frame_idx):

    source, yolo_weights, deep_sort_weights, show_vid, save_vid, save_txt, imgsz, evaluate, nosave, project, name, exist_ok = \
        opt.source, opt.yolo_weights, opt.deep_sort_weights, opt.show_vid, opt.save_vid, \
        opt.save_txt, opt.img_size, opt.evaluate, opt.nosave, opt.project, opt.name, opt.exist_ok
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')
    # Load model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16
    pred = model(img, augment=opt.augment)[0]

    # Apply NMS
    pred = non_max_suppression(
        pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    #outputs=[]
    for i, det in enumerate(pred):  # detections per image
        outputs = []
        if webcam:  # batch_size >= 1
            p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
        else:
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

        # s += '%gx%g ' % img.shape[2:]  # print string
        # save_path = str(Path(out) / Path(p).name)

        p = Path(p)  # to Path
        save_path = str(save_dir / p.name)  # im.jpg
        save_path_img = str(save_dir / p.stem)
        (save_dir / p.stem).mkdir(parents=True, exist_ok=True)
        txt_path = str(save_dir / 'labels' / p.stem)+'-tooth' + '.txt'  # im.txt
        s += '%gx%g ' % img.shape[2:]
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

            xyxy = det[:, 0:4].cpu().numpy()
            print('xyxy=',xyxy)
            confs = det[:, 4].cpu().numpy()
            clss = det[:, 5].cpu().numpy()
            print('clss=',clss)

            if len(xyxy) > 0:
                for j, (output,cls, conf) in enumerate(zip(xyxy, clss, confs)):
                    #print('output=',output)
                    bboxes = output[0:4]
                    #print('len=',len(output))
                    #print('bboxes=', bboxes)

                    #print('output=', output)

                    c = int(cls)  # integer class
                    label = f'{names[c]} {conf:.2f}'
                    #print('label=', label)
                    a = numpy.append(output,cls)
                    #print('a=', a)
                    outputs.append(a)

                    if save_txt:
                        # to MOT format
                        bbox_top = output[0]
                        bbox_left = output[1]
                        bbox_w = output[2]
                        bbox_h = output[3]
                        #write tooth boxx
                        id=frame_idx
                        #color = compute_color_for_id(id)
                        #plot_one_box(output, im0, label=None, color=color, line_thickness=2)
                        #save_image_dir = os.path.join(save_path_img, '%s.jpg' % frame)
                        # x=x+1
                        #print('save_image_dir: ', save_image_dir)
                        #cv2.imwrite(save_image_dir, im0)

                        # Write MOT compliant results to file
                        with open(txt_path, 'a') as f:

                                f.write(('%g ' + '%s ' + '%g ' * 4 + '\n') % (frame_idx, names[c], bbox_top,
                                                                                  bbox_left, bbox_w,
                                                                                  bbox_h))  # label format

    return outputs
