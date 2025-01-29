import cv2
import torch
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import json
from torch import Tensor,optim,nn
from torchvision import transforms
from model.yolox import YOLOX
from dataset.dataset import create_dataset
from model.utils import preproc

#模型初始化
model = torch.load("RAOD-2000.pth")
model.cuda(0)

input_size = 1280
epoch = 100
batch_size = 4
lr = 0.01
optimizer = optim.SGD(model.parameters(), lr=lr,weight_decay=5e-3,momentum=0.9)

#数据集
dataset = create_dataset(True,batch_size)
def load_raw(name):
    BIT8, BIT16, BIT24 = 2 ** 8, 2 ** 16, 2 ** 24
    a = []
    for i in range(len(name)):
        fn = name[i]
        raw = np.fromfile(fn, dtype=np.uint8)
        raw = raw.reshape(1856, 2880, 3).astype(np.float32)
        raw = np.split(raw, 3, axis=2)
        raw = (raw[0] + raw[1] * BIT8 + raw[2] * BIT16)
        raw_data = raw.transpose(2, 0, 1)
        raw_data = np.expand_dims(raw_data, axis=0)
        raw_data = Tensor(raw_data).cuda()
        demosaic = Demosaic()
        demosaic.cuda()
        rgb_hdr_data = demosaic(raw_data)
        # print(rgb_hdr_data)
        AWB = GrayWorldWB()
        rgb_hdr_data = AWB(rgb_hdr_data)
        rgb_hdr_data = rgb_hdr_data.cpu()
        rgb_hdr_data = np.array(rgb_hdr_data)
        bgr_hdr_data = rgb_hdr_data[:, ::-1, :, :]
        bgr_hdr_data = bgr_hdr_data.squeeze()
        bgr_hdr_data = bgr_hdr_data.transpose(1, 2, 0)
        tonemap = cv2.createTonemapReinhard(gamma=3.1, intensity=-5, light_adapt=0.2, color_adapt=0.0)
        bgr_ldr_img = tonemap.process(bgr_hdr_data)
        bgr_ldr_img = np.clip(bgr_ldr_img * 255, 0, 255).astype('uint8')
        result, encoded_image = cv2.imencode('.jpg', bgr_ldr_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        nparr = np.frombuffer(encoded_image, np.uint8)
        bgr_ldr_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        rgb_ldr_img = bgr_ldr_img[:, :, ::-1]
        d, _ = preproc(rgb_ldr_img, (1280, 1280))
        d = torch.from_numpy(d)
        d = d.float()
        a.append(d)
    a = np.array(a)
    return Tensor(a)
def load_anno(name):
    a = []
    for i in range(len(name)):
        fn = name[i]
        fn = f"/mnt/pycharm/datasets/anno/{fn[27:]}"
        fn = f"{fn[:-4]}.json"
        po = []
        i = 0
        with open(fn) as f: data = json.load(f)
        for obj in data['shapes']:
            i = i+1
            bo = []
            klass = obj['label']  # object class
            if klass == 'Pedestrian':
                klass = 0
            elif klass == 'Cyclist':
                klass = 1
            elif klass == 'Car':
                klass = 2
            elif klass == 'Truck':
                klass = 3
            elif klass == 'Tram':
                klass = 4
            bo.append(klass)
            xmin, ymin = obj['points'][0]  # bounding box
            xmax, ymax = obj['points'][1]  # bounding box
            assert 0 <= xmin < xmax <= 2880
            assert 0 <= ymin < ymax <= 1856
            bo.append((input_size*xmin/2880+input_size*xmax/2880)/2)
            bo.append((input_size*ymin/2880+input_size*ymax/2880)/2)
            bo.append(input_size*xmax/2880-input_size*xmin/2880)
            bo.append(input_size*ymax/2880-input_size*ymin/2880)
            bo = np.array(bo)
            po.append(bo)
        for i in range(len(po), 70):
            bo = np.array([0,0,0,0,0])
            po.append(bo)
        po = np.array(po)
        a.append(po)
    a = np.array(a)
    return Tensor(a)


for epochs in range(45,epoch):
    print(f"-----第{epochs+1}轮训练-----")
    if (epochs+1) < 25:
        for param in model.backbone.parameters():
            param.requires_grad = False
        if (epochs + 1) < 10:
            lr = (epochs + 1) * 0.001
        else:
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=0, last_epoch=-1)
    elif 25 <=(epochs+1) < 45:
        for param in model.backbone.parameters():
            param.requires_grad = True
        for param in model.tmm.parameters():
            param.requires_grad = False
        if (epochs + 1) < 35:
            lr = (epochs + 1) * 0.01
        else:
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=0, last_epoch=-1)
    else:
        for param in model.parameters():
            param.requires_grad = True
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 15, eta_min=0, last_epoch=-1)

    for step,name in enumerate(dataset):

        model.train()
        img = load_raw(name)
        labels = load_anno(name)
        img = img.cuda()
        img = trans(img)
        labels = labels.cuda()
        loss = model(img, labels)
        if (epochs<45):
            nn.utils.clip_grad_norm_(model.parameters(), 0.3)
        optimizer.zero_grad()
        loss["total_loss"].backward()
        optimizer.step()

        if (step+1)%50==0:
            print("total_loss:",end="")
            print(loss["total_loss"].item(),end="   ")
            print("iou_losss:", end="")
            print(loss["iou_loss"].item(), end="   ")
            print("cls_loss:", end="")
            print(loss["cls_loss"].item())
        if (step+1)%1000==0 :
            torch.save(model, f"RAOD-{step + 1}.pth")
    torch.save(model,f"RAOD-{epochs+1}.pth")
    torch.cuda.empty_cache()