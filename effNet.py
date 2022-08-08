import os
import json
import cv2
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import efficientnet_b0 as create_model


def effNet(img_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_size = {"B0": 224,
                "B1": 240,
                "B2": 260,
                "B3": 300,
                "B4": 380,
                "B5": 456,
                "B6": 528,
                "B7": 600}
    num_model = "B0"
    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = create_model(num_classes=2).to(device)
    # load model weights
    model_weight_path = "./model-99.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    data_transform = transforms.Compose(
        [transforms.Resize(img_size[num_model]),
         transforms.CenterCrop(img_size[num_model]),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    os.makedirs(img_path+'tooth/',exist_ok=True)
    folderlist = os.listdir(img_path)
    for filename in folderlist:
        torch.cuda.empty_cache()
        import time
        start=time.time()
        # load image
        #img_path = "E:/ljf/efficientNet/detect_imgs/00001.jpg"
        #assert os.path.exists(img_path + filename), "file: '{}' dose not exist.".format(filename)
        Img = Image.open(img_path+'/' + filename)
        plt.imshow(Img)
        # [N, C, H, W]
        img = data_transform(Img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)
        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
        if (class_indict[str(predict_cla)] == 'tooth'):
            Img.save(img_path+'tooth/'+filename)
        print(time.time()-start)
    return img_path+'tooth'

def effNet_img(img):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    img_size = {"B0": 224,
                "B1": 240,
                "B2": 260,
                "B3": 300,
                "B4": 380,
                "B5": 456,
                "B6": 528,
                "B7": 600}
    num_model = "B0"

    data_transform = transforms.Compose(
        [transforms.Resize(img_size[num_model]),
         transforms.CenterCrop(img_size[num_model]),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = create_model(num_classes=2).to(device)
    # load model weights
    model_weight_path = "./model-99.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
    return predict_cla
