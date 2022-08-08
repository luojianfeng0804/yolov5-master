from models.experimental import attempt_load
from utils.torch_utils import select_device
from PIL import Image
import base64
import io
from flask import Flask, request, jsonify,render_template

import json
import numpy as np
from backend.predict import predict
from pathlib import Path
from models.common import DetectMultiBackend
import cv2
import os
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
# 传入__name__实例化Flask
app = Flask(__name__)
app.debug = True  # Flask内置了调试模式，可以自动重载代码并显示调试信息
app.debug = True  # Flask内置了调试模式，可以自动重载代码并显示调试信息
# 读取flask配置
with open('./backend/flask_config.json','r',encoding='utf8')as fp:
    opt = json.load(fp)
    print('Flask Config : ', opt)

# 选择设备
device = select_device("0")
# 加载模型
#model = attempt_load(opt['weights'], map_location=device)
model = attempt_load('weights-qc/best.pt', map_location=device)
model0 = attempt_load('weights-qc/tooth_best.pt', map_location=device)
@app.route('/predict/', methods=['POST'])

# 响应POST消息的预测函数
def get_prediction():
    import time
    from Ubuntu_video import smooth_video
    from effNet import effNet
    imgdata = []
    start =time.time()
    f = request.files.get('pic')
    fname = f.filename
    print(fname)
    file_path = os.path.join(opt['source'], f.filename)  # filename是f的固有属性
    f.save(file_path)
    #LoadImages(file_path,img_size=opt['imgsz'])
    path1 = smooth_video(file_path)
    #print(path1)
    msg = 'no decayed tooth'
    if len(os.listdir(path1))==0:
        print(1)
        return jsonify({"code":0,"msg":msg,"data":imgdata})
    path1 = effNet(path1)
    if len(os.listdir(path1))==0:
        print(2)
        return jsonify({"code":0,"msg":msg,"data":imgdata})
    fname = fname[:-4]    
    imgdata = predict(opt, model,dataset, fname) # 预测图像
    end = time.time()
    print('Running time1: %s Seconds'%(end1-start))
    print('trans time: %s Seconds'%(end2-end1))
    print('Running time: %s Seconds'%(end-start))
    if len(imgdata)==0 :
        msg = 'no decayed tooth'
    else :
        msg = 'have decayed tooth'
    return jsonify({"code":0,"msg":msg,"data":imgdata})

@app.after_request
def add_headers(response):
    # 允许跨域
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    return response
def return_img_stream(img_local_path):
    import base64
    imgdata = []
    img_stream = ''
    for img in os.listdir(img_local_path):
        img_stream = img.read()
        img_stream = base64.b64encode(img_stream).decode()
        imgdata.append(img_stream)
    return img_stream

@app.route('/index')
def hello_world():
    img_path = 'static/output/'
    img_stream_ = return_img_stream(img_path)
    return render_template('index.html',img_stream=img_stream_)

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1',port=5000)
    #app.run(debug=False, host='127.0.0.1')

@app.route('/uploads/<path:filename>')
def get_file(filename):
    return send_from_directory(app.config['UPLOAD_PATH', filename])

