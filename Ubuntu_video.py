# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as Ssim
import shutil
import time

def yidong(filename1, filename2):
    shutil.move(filename1, filename2)


def delete(filename1):
    os.remove(filename1)


# 相似度，如果大于max_ssim就进行删除
max_ssim = 0.72


# 利用拉普拉斯算子计算图片的二阶导数，反映图片的边缘信息，同样事物的图片，
# 清晰度高的，相对应的经过拉普拉斯算子滤波后的图片的方差也就越大。
def getImageVar(image):
    img2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageVar = cv2.Laplacian(img2gray, cv2.CV_64F).var()
    return imageVar


def smooth_video(video_path):
    frame_diffs = []
    img_files = []
    var_frame = []
    start = time.perf_counter()
    folder_name = video_path.split('.')[0]
    # folder_name = video_path + video_path.split('/')[-1]
    os.makedirs(folder_name, exist_ok=True)  # 创建目录
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    i = 0
    while (ret):
        if (i % 5 == 0):
            img_files.append(frame)
        ret, frame = cap.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        i = i + 1
    print('img', len(img_files))

    for frame in img_files:
        var_image = getImageVar(frame)
        var_frame.append(var_image)

    currIndex = 0
    currIndex1 = currIndex + 1
    i = 0
    while (currIndex1 < len(img_files) - 1):
        i = i + 1
        #print('currIndex',currIndex)
        #print('currIndex1',currIndex1)
        img = img_files[currIndex]
        img1 = img_files[currIndex1]
        ssim = Ssim(img, img1, multichannel=True)
        if ssim > max_ssim:
            # 计算图像清晰度
            # var_img = getImageVar(img)
            # 计算图像清晰度
            # var_img1 = getImageVar(img1)
            # 当达到了门限，比较两个图像的清晰度，把不清晰的放入删除列表中
            if (var_frame[currIndex] > var_frame[currIndex1]):
                currIndex = currIndex
                currIndex1 = currIndex1 + 1
            else:
                currIndex = currIndex1
                currIndex1 = currIndex1 + 1
        else:
            frame_diffs.append(currIndex)
            currIndex = currIndex1
            currIndex1 = currIndex1 + 1
        # print(len(img_files))
    print(i)
    print(len(frame_diffs))
    pic_path = folder_name + '/'
    for i in frame_diffs:
        name = "frame_" + str(i) + ".jpg"
        cv2.imwrite(pic_path+'_' + name, img_files[i])
    end = time.perf_counter()
    print('time',end - start)
    return folder_name





