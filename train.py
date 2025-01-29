import torch
import warnings
warnings.filterwarnings("ignore")
from torch import optim,nn
from model.yolox import YOLOX
from dataset.dataset import create_dataset,change
from tqdm import *
from dataset.dataset import load_anno,load_raw
from dataset.demosaic import rgbraw
from torch import Tensor
from eval import val_
from torch.backends import cudnn
from dataset.mosaic import augment_batch
torch.manual_seed(1)
input_size = 512
epoch = 100
batch_size = 16
dataset = create_dataset(True,batch_size)
torch.autograd.set_detect_anomaly(True)
def main():
    # 模型初始
    model = YOLOX()
    def weights_init_(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal(m.weight)
            if m.bias is not None:
                nn.init.constant(m.bias, 1e-2)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform(m.weight)
            if m.bias is not None:
                nn.init.constant(m.bias, 1e-2)
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal(m.weight)
            if m.bias is not None:
                nn.init.constant(m.bias, 1e-2)
        elif isinstance(m, nn.Linear):
            nn.init.uniform(m.weight)
            if m.bias is not None:
                nn.init.constant(m.bias, 1e-2)
    #model.apply(weights_init_)
    cudnn.benchmark = True
    cudnn.enabled = True
    model.load_state_dict(torch.load("best_.pth"), strict=False)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    best_ap = 0.0
    model.cuda()
    for epochs in range(epoch):
        for step,name in enumerate(tqdm(dataset,desc=f"Training Epoch{epochs+1}")):
            lr = 0.02
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,weight_decay=0.0005)
            optim.lr_scheduler.CosineAnnealingLR(optimizer,200)
            model.train()
            img = load_raw(name,input_size)
            labels = load_anno(name, input_size)
            #img, labels = augment_batch(img, labels, (input_size,input_size))
            torch.cuda.empty_cache()
            labels = change(labels)
            img =Tensor(img).cuda()
            labels =Tensor(labels).cuda()
            img =  torch.clamp(img, 0, 1)
            from torch.cuda.amp import autocast
            with autocast(True):
                loss = model(img,labels)
            optimizer.zero_grad()
            loss["total_loss"].backward()
            optimizer.step()
            torch.cuda.empty_cache()
            if (step+1)%100==0:
                print("total_loss:",end="")
                print(loss["total_loss"].item(),end="   ")
                print("iou_losss:", end="")
                print(loss["iou_loss"].item(), end="   ")
                print("cls_loss:", end="")
                print(loss["cls_loss"].item())
        ap = val_(model)
        if ap>=best_ap:
            best_ap = ap
            torch.save(model,f"best_{ap}.pth")

if __name__ == "__main__":
    main()