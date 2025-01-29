import glob
import os
import numpy as np
import torch
import cv2
from .demosaic import  demosaic,GrayWorldWB
from torch.utils.data import Dataset, DataLoader
import rawpy
import xml.etree.ElementTree as ET


class dataset(Dataset):
    def __init__(self,is_train):
        path = 'C:/Users/luzhiyi/Desktop/jpg'
        self.flist = glob.glob(os.path.join(path, '*.jpg'))
        d = []
        if is_train==False:
            for i in range(len(self.flist)):
                if self.flist[i][-6]=='6':
                    d.append(self.flist[i])
        else:
            for i in range(len(self.flist)):
                if self.flist[i][-6]!='6':
                    d.append(self.flist[i])
        self.flist = d
        self.length = len(self.flist)
        print(f'{self.length} raw images found')

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        idx = index % self.length
        fn = self.flist[idx]
        return fn

def create_dataset(is_train,batch_size):
    datasets = dataset(is_train)
    data = DataLoader(datasets, batch_size=batch_size,pin_memory=True,shuffle=True)
    return data

def create_dataset_val(is_train,batch_size):
    datasets = dataset(is_train)
    data = DataLoader(datasets, batch_size=batch_size,pin_memory=True,shuffle=is_train)
    return data

def load_anno(name,input_size):
    a = []
    for i in range(len(name)):
        fn = name[i]
        fn =f"L:/anno/{fn[28:]}"
        fn = f"{fn[:-4]}.xml"
        tree = ET.parse(fn)
        root = tree.getroot()
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
            # 边界框两个坐标点
            xmin = int(float(xmlbox.find("xmin").text))
            ymin = int(float(xmlbox.find("ymin").text))
            xmax = int(float(xmlbox.find("xmax").text))
            ymax = int(float(xmlbox.find("ymax").text))
            bo.append(klass)
            bo.append(input_size * xmin / 600)
            bo.append(input_size * ymax / 400)
            bo.append(input_size * xmax / 600)
            bo.append(input_size * ymin / 400)
            bo = np.array(bo)
            po.append(bo)
        po = np.array(po)
        a.append(po)
    return a

def load_raw(name,input_size):
    a = []
    #AWB = GrayWorldWB()
    for i in range(len(name)):
        fn = name[i]
        '''raw = rawpy.imread(fn)
        im = raw.raw_image.astype(np.float32)
        raw_data = np.expand_dims(np.array(im),axis=0)
        rgb_hdr_data = demosaic(raw_data)
        rgb_hdr_data = torch.from_numpy(rgb_hdr_data).unsqueeze(0)
        rgb_hdr_data = AWB(rgb_hdr_data)
        rgb_hdr_data = rgb_hdr_data.squeeze(0).detach().numpy()
        bgr_hdr_data = rgb_hdr_data[::-1, :, :]
        bgr_hdr_data = bgr_hdr_data.transpose(1, 2, 0)
        tonemap = cv2.createTonemapReinhard(gamma=3.1, intensity=-5, light_adapt=0.2, color_adapt=0.0)
        bgr_ldr_img = tonemap.process(bgr_hdr_data)
        bgr_ldr_img = np.clip(bgr_ldr_img * 255, 0, 255).astype('uint8')
        bgr_ldr_img = cv2.resize(bgr_ldr_img, (input_size, input_size))
        rgb_ldr_img = bgr_ldr_img[:, :, ::-1]
        rgb_ldr_img = rgb_ldr_img.transpose(2, 0, 1)'''
        rgb_ldr_img = cv2.imread(fn)
        rgb_ldr_img = cv2.resize(rgb_ldr_img, (input_size, input_size))
        rgb_ldr_img = rgb_ldr_img.transpose(2, 0, 1)
        a.append(rgb_ldr_img/255.0)
    return np.array(a)

def change(label):
    for i in range(len(label)):
        a = label[i]
        for j in range(a.shape[0]):
            b = a[j]
            xmin = b[1]
            ymax = b[2]
            xmax = b[3]
            ymin = b[4]
            label[i][j][1] = (xmin+xmax)/2
            label[i][j][2] = (ymin+ymax)/2
            label[i][j][3] = xmax - xmin
            label[i][j][4] = ymax - ymin
        if label[i].shape[0] == 0:
            label[i] = np.array([[0, 0, 0, 0, 0]])
        for j in range(label[i].shape[0],70):
            bo = np.array([[0, 0, 0, 0, 0]])
            label[i] = np.concatenate((label[i], bo), axis=0)
    label = np.array(label)
    return label

