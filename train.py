import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")
import os
from torch import optim,nn
from nets.frcnn import FasterRCNN
from dataset.dataset import create_dataset
from nets.frcnn_training import FasterRCNNTrainer,weights_init
from tqdm import *
from dataset.dataset import load_anno,load_raw,change
from dataset.mosaic import augment_batch
from torch import Tensor
from eval import val_
from torch.backends import cudnn
torch.manual_seed(1)
input_size = 512
epoch = 100
batch_size = 12
dataset = create_dataset(True,batch_size)
torch.autograd.set_detect_anomaly(True)
def main():
    # 模型初始
    model = FasterRCNN(3,anchor_scales=[8, 16, 32],backbone="resnet50",pretrained=True)
    cudnn.benchmark = True
    cudnn.enabled = True
    best_ap = 0.0
    best_ap1 = 0.0
    weights_init(model)
    model.cuda()
    for epochs in range(2,epoch):
        if epoch < 2 :
            lr = 0.001
        else:
            lr = 0.01
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        train_util = FasterRCNNTrainer(model, optimizer)
        for step,name in enumerate(tqdm(dataset,desc=f"Training Epoch{epochs+1}")):
            optim.lr_scheduler.CosineAnnealingLR(optimizer, 200)
            img = load_raw(name,input_size)
            labels = load_anno(name, input_size)
            #img, labels = augment_batch(img, labels, (input_size, input_size))
            boxes, label = change(labels)
            torch.cuda.empty_cache()
            img =Tensor(img).cuda()
            img =  torch.clamp(img, 0, 1)
            rpn_loc, rpn_cls, roi_loc, roi_cls, total = train_util.train_step(img, boxes, label, 1, True,None)
            torch.cuda.empty_cache()
            if (step+1)%100==0:
                print("total_loss:",end="")
                print(total.item(),end="   ")
                print("iou_losss:", end="")
                print(rpn_cls.item(), end="   ")
                print("cls_loss:", end="")
                print(rpn_cls.item())
            torch.save(model.state_dict(),f"best.pth")
        apa = val_()
        if apa > best_ap:
            best_ap = apa
            torch.save(model.state_dict(),f"best_{apa}.pth")

if __name__ == "__main__":
    main()
