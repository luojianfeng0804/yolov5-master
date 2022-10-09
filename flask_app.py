
from PIL import Image
import base64
import io
from flask import Flask, request, jsonify,render_template
import json
import numpy as np
import cv2
import os
from yolov5.utils.torch_utils import select_device
from yolov5.models.experimental import attempt_load

# 传入__name__实例化Flask
app = Flask(__name__)
app.debug = True  # Flask内置了调试模式，可以自动重载代码并显示调试信息
app.debug = True  # Flask内置了调试模式，可以自动重载代码并显示调试信息
# 读取flask配置
from track_test import yolo_predict

# 选择设备
device = select_device("0")
# 加载模型
#model = attempt_load(opt['weights'], map_location=device)
model = attempt_load('weights/best.pt', map_location=device)
model0 = attempt_load('weights/tooth_best.pt', map_location=device)
ry_model = attempt_load('weights/ry_best.pt', map_location=device)

@app.route('/predict/', methods=['POST'])

# 响应POST消息的预测函数
def get_prediction():
    import time
    imgdata = []
    start = time.time()
    type_int = request.args.get("type")
    # f = request.files.get('pic')
    # fname = f.filename
    # print(fname)
    # file_path = os.path.join('source', f.filename)  # filename是f的固有属性
    # print(file_path)
    # f.save(file_path)
    #img_path = ""
    file_path = request.args.get('url')
    if type_int == 0:
        img_path = yolo_predict(file_path, model, model0)  # 预测图像
    else:
        img_path = yolo_predict(file_path, ry_model, model0)  # 预测图像
    print('f')
    #fname = fname[:-4]
    int_name = []
    folderlist = os.listdir(img_path)
    for filename in folderlist:
        filename = int(filename[:-4])
        int_name.append(filename)
    int_name.sort()
    for filename in int_name:
        print(filename)
        imgdata.append('http://101.201.208.143/'+img_path+'/'+str(filename)+'.jpg')
    end = time.time()
    print('Running time: %s Seconds'%(end-start))
    if len(imgdata) == 0 :
        msg = 'no decayed tooth'
    else:
        msg = 'have decayed tooth'
    print(type_int)
    return jsonify({"code": type_int, "msg": msg, "data": imgdata})

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

