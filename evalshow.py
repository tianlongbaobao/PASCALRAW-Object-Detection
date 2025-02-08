import numpy as np
from torch import Tensor
import torch
import cv2
import warnings
warnings.filterwarnings("ignore")
from model.utils import postprocess
from dataset.demosaic import GrayWorldWB,Demosaic
from nets.frcnn import FasterRCNN
from utils.utils_bbox import DecodeBox

input_size = 512
model = FasterRCNN(3,anchor_scales=[8, 16, 32],backbone="resnet50",pretrained=False,mode="predict")
model.load_state_dict(torch.load("best.pth"))
model.cuda()
model.eval()

def load_raw(name,input_size):
    fn = name
    rgb_ldr_img = cv2.imread(fn)
    rgb_ldr_img = cv2.resize(rgb_ldr_img, (input_size, input_size))
    rgb_ldr_img = rgb_ldr_img.transpose(2, 0, 1)
    return Tensor(rgb_ldr_img/255.0)

def load_raw_show(name,input_size):
    fn = name
    rgb_ldr_img = cv2.imread(fn)
    rgb_ldr_img = cv2.resize(rgb_ldr_img, (input_size, input_size))
    return rgb_ldr_img
def show(img):
    cv2.namedWindow("image",cv2.WINDOW_NORMAL)
    cv2.imshow("image",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

path = "C:/Users/luzhiyi/Desktop/jpg/2014_000006.jpg"

img_s = load_raw_show(path,input_size)
img = load_raw(path,input_size)
img = img.unsqueeze(0).cuda()
bbox_util = DecodeBox(torch.tensor([0.1, 0.1, 0.2, 0.2],device="cuda").repeat(3 + 1)[None], 3)
with torch.no_grad():
    roi_cls_locs, roi_scores, rois, roi_indices = model(img)
results = bbox_util.forward(roi_cls_locs, roi_scores, rois, [512,512], (512, 512),
                                     nms_iou=0.5, confidence=0.3)
train_label = np.array(results[0][:, 5], dtype='int32')
train_conf = results[0][:, 4]
train_reg = results[0][:, :4]
print(train_reg)
for i in train_reg:
    cv2.rectangle(img_s,(int(i[1]),int(i[0])),(int(i[3]),int(i[2])),(255,0,0),2)
show(img_s)
