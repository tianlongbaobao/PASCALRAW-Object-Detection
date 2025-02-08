import torch
import cv2
from torch import  Tensor
import numpy as np
import xml.etree.ElementTree as ET
from nets.frcnn import FasterRCNN

from tqdm import *
import warnings
from utils.utils_bbox import DecodeBox
warnings.filterwarnings("ignore")
from model.utils import postprocess,bboxes_iou
from dataset.dataset import create_dataset_val
torch.manual_seed(1)
input_size = 512
def load_raw(name,input_size):
    rgb_ldr_img = cv2.imread(name)
    rgb_ldr_img = cv2.resize(rgb_ldr_img, (input_size, input_size))
    rgb_ldr_img = rgb_ldr_img.transpose(2, 0, 1)
    return rgb_ldr_img/255.0

def load_anno(name):
    fn = name
    fn = f"L:/anno/{fn[28:]}"
    fn = f"{fn[:-4]}.xml"
    tree = ET.parse(fn)
    root = tree.getroot()
    b = []
    po = []
    for obj in root.iter("object"):
        bo = []
        cls = obj.find("name").text
        if cls == 'person':
            klass = 0
        elif cls == 'bicycle':
            klass = 1
        elif cls == 'car':
            klass = 2
        xmlbox = obj.find("bndbox")
        xmin = int(float(xmlbox.find("xmin").text))
        ymin = int(float(xmlbox.find("ymin").text))
        xmax = int(float(xmlbox.find("xmax").text))
        ymax = int(float(xmlbox.find("ymax").text))
        b.append(klass)
        bo.append(input_size * xmin / 600)
        bo.append(input_size * ymax / 400)
        bo.append(input_size * xmax / 600)
        bo.append(input_size * ymin / 400)
        bo = np.array(bo)
        po.append(bo)
    po = np.array(po)
    b = np.array(b)
    return Tensor(po),Tensor(b)

def show(img):
    cv2.namedWindow("image",cv2.WINDOW_NORMAL)
    cv2.imshow("image",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def classify(model,img,val_reg,val_label,all,zero,one,two,zeroc,onec,twoc):
    model.eval()
    bbox_util = DecodeBox(torch.tensor([0.1, 0.1, 0.2, 0.2],device="cuda").repeat(3 + 1)[None], 3)
    with torch.no_grad():
        roi_cls_locs, roi_scores, rois, roi_indices = model(img)
    results = bbox_util.forward(roi_cls_locs, roi_scores, rois, [512,512], (512, 512),
                                     nms_iou=0.5, confidence=0.1)
    train_label = np.array(results[0][:, 5], dtype='int32')
    train_conf = results[0][:, 4]
    train_reg = results[0][:, :4]
    for i in val_label:
        if int(i.item()) == 0:
            zeroc += 1
        elif int(i.item()) == 1:
            onec += 1
        elif int(i.item()) == 2:
            twoc += 1
    val_reg = val_reg[:,[0,3,2,1]]
    train_reg = train_reg[:,[1,0,3,2]]
    iou = bboxes_iou(Tensor(val_reg), Tensor(train_reg), True)
    for i in range((iou.shape)[0]):
        maxiou = iou[i][0]
        klass = val_label[i]
        conf = train_conf[0]
        if klass == train_label[0]:
            judge = 1
        else:
            judge = 0
        for j in range((iou.shape)[1]):
            if maxiou<iou[i][j]:
                conf = train_conf[j]
                maxiou = iou[i][j]
                if klass == train_label[j]:
                    judge = 1
                else:
                    judge = 0
        b = []
        b.append(maxiou)
        b.append(judge)
        b.append(conf)
        all.append(b)
        if klass == 0:
            zero.append(b)
        elif klass == 1:
            one.append(b)
        elif klass == 2:
            two.append(b)
    return zeroc,onec,twoc

def test(result,sam,b):
    result = sorted(result,key=lambda x:(x[2].item()),reverse=True)
    tp = 0
    recall = []
    pre = 0
    precision = []
    for i in range(len(result)):
        if result[i][0] > b and result[i][1] == 1:
            tp += 1
        pre = pre + 1
        precision.append(tp/pre)
        recall.append(tp/sam)
    ap = np.trapz(precision,recall)+(1-recall[-1])*precision[-1]/2
    return ap
def val_():
    model = FasterRCNN(3,anchor_scales=[8, 16, 32],backbone="resnet50",pretrained=False,mode="predict")
    model.load_state_dict(torch.load("best_0.9254717891560675.pth"))
    model.cuda()
    model.eval()
    input_size = 512
    val = create_dataset_val(False,1)
    all = []
    zero = []
    one = []
    two = []
    zeroc = 0
    onec = 0
    twoc = 0
    for step, path in enumerate(tqdm(val, desc="Validation")):
        img = load_raw(path[0],input_size)
        img = torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0)
        img = img.cuda()
        img = img.float()
        val_reg,val_label = load_anno(path[0])
        if  np.array(val_reg).shape[0] != int(0):
            try:
                zeroc,onec,twoc=classify(model,img,val_reg,val_label,all,zero,one,two,zeroc,onec,twoc)
            except:
                pass
    allc = zeroc+onec+twoc
    print(f"person:{zeroc},bicycle:{onec},car:{twoc},All:{allc}")
    ap0 = test(zero, zeroc, 0.5)
    ap1 = test(one, onec, 0.5)
    ap2 = test(two, twoc, 0.5)
    apa = test(all, allc, 0.5)
    ap01 = test(zero, zeroc, 0.75)
    ap12 = test(one, onec, 0.75)
    ap23 = test(two, twoc, 0.75)
    map75 = (ap01 + ap12 + ap23) / 3
    apaa = test(all, allc, 0.75)
    apa1 = test(all, allc, 0.6)
    apa2 = test(all, allc, 0.7)
    apa3 = test(all, allc, 0.8)
    apa4 = test(all, allc, 0.9)
    map = (apa + apa1 + apa2 + apa3 + apa4) / 5
    print(f"AP50\nPedestrian:{ap0}\nCyclist:{ap1}\nCar:{ap2}\nAll:{apa}\nAP75:{apaa}\nmap:{map}\nmap75:{map75}")
    return apa

val_()