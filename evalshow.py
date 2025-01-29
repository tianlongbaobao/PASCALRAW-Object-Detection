import numpy as np
from torch import Tensor
import torch
import cv2
import warnings
warnings.filterwarnings("ignore")
from model.utils import postprocess
from dataset.demosaic import GrayWorldWB,Demosaic
from model.yolox import YOLOX

input_size = 960
model = YOLOX()
torch.manual_seed(0)
model = torch.load("best_0.9113741187023321.pth")
model.eval()
model.cuda()

def load_raw(name,input_size):
    BIT8, BIT16, BIT24 = 2 ** 8, 2 ** 16, 2 ** 24
    fn = name
    raw = np.fromfile(fn, dtype=np.uint8)
    raw = raw.reshape(1856, 2880, 3).astype(np.float32)
    raw = np.split(raw, 3, axis=2)
    raw = (raw[0] + raw[1] * BIT8 + raw[2] * BIT16)
    raw_data = raw.transpose(2, 0, 1)
    raw_data = np.expand_dims(raw_data, axis=0)
    raw_data = Tensor(raw_data).cuda()
    demosaic = Demosaic()
    rgb_hdr_data = demosaic(raw_data)
    rgb_hdr_data = rgb_hdr_data.cpu()
    rgb_hdr_data = np.array(rgb_hdr_data)
    bgr_hdr_data = rgb_hdr_data[:, ::-1, :, :]
    bgr_hdr_data = bgr_hdr_data.squeeze()
    bgr_hdr_data = bgr_hdr_data.transpose(1, 2, 0)
    bgr_ldr_img = bgr_hdr_data / (BIT24 - 1)
    rgb_ldr_img = cv2.resize(bgr_ldr_img, (input_size, input_size))
    rgb_ldr_img = rgb_ldr_img.transpose(2, 0, 1)
    return Tensor(rgb_ldr_img)

def load_raw_show(name,input_size):
    BIT8, BIT16, BIT24 = 2 ** 8, 2 ** 16, 2 ** 24
    fn = name
    raw = np.fromfile(fn, dtype=np.uint8)
    raw = raw.reshape(1856, 2880, 3).astype(np.float32)
    raw = np.split(raw, 3, axis=2)
    raw = (raw[0] + raw[1] * BIT8 + raw[2] * BIT16)
    raw_data = raw.transpose(2, 0, 1)
    raw_data = np.expand_dims(raw_data, axis=0)
    raw_data = Tensor(raw_data).cuda()
    demosaic = Demosaic()
    rgb_hdr_data = demosaic(raw_data)
    # print(rgb_hdr_data)
    AWB = GrayWorldWB()
    rgb_hdr_data = AWB(rgb_hdr_data)
    rgb_hdr_data = rgb_hdr_data.cpu().numpy()
    rgb_hdr_data = np.array(rgb_hdr_data)
    bgr_hdr_data = rgb_hdr_data[:, ::-1, :, :]
    bgr_hdr_data = bgr_hdr_data.squeeze()
    bgr_hdr_data = bgr_hdr_data.transpose(1, 2, 0)
    tonemap = cv2.createTonemapReinhard(gamma=3.1, intensity=-5, light_adapt=0.2, color_adapt=0.0)
    bgr_ldr_img = tonemap.process(bgr_hdr_data)
    bgr_ldr_img = np.clip(bgr_ldr_img * 255, 0, 255).astype('uint8')
    result, encoded_image = cv2.imencode('.jpg', bgr_ldr_img, [int(cv2.IMWRITE_JPEG_QUALITY), 99])
    nparr = np.frombuffer(encoded_image, np.uint8)
    bgr_ldr_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    #rgb_ldr_img = bgr_ldr_img[:, :, ::-1]
    rgb_ldr_img = cv2.resize(bgr_ldr_img, (input_size, input_size))
    return rgb_ldr_img
def show(img):
    cv2.namedWindow("image",cv2.WINDOW_NORMAL)
    cv2.imshow("image",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

path = "C:/Users/luzhiyi/Desktop/train/day-02528.raw"

img_s = load_raw_show(path,input_size)
img = load_raw(path,input_size)
img = img.unsqueeze(0).cuda()
outputs,imgout = model(img)
imgout = imgout.detach().cpu().numpy()
result = postprocess(outputs,5,conf_thre=0.01,nms_thre=0.45)
result = result[0]
for i in range(len(result)):
    o = result[i]
    if o[6] == 2:
        cv2.rectangle(img_s,(int(o[0]),int(o[1])),(int(o[2]),int(o[3])),(255,0,0),2)
show(img_s)
show(imgout[0].transpose(1, 2, 0))
