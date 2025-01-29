import numpy as np
import cv2
import torch
import warnings
warnings.filterwarnings("ignore")
import json
from torch import Tensor
from dataset.demosaic import GrayWorldWB,Demosaic
from model.utils import postprocess
def eval_loss(model,input_size):
    def load_raw(name):
        BIT8, BIT16, BIT24 = 2 ** 8, 2 ** 16, 2 ** 24
        fn = name
        raw = np.fromfile(fn, dtype=np.uint8)
        raw = raw.reshape(1856, 2880, 3).astype(np.float32)
        raw = np.split(raw, 3, axis=2)
        raw = (raw[0] + raw[1] * BIT8 + raw[2] * BIT16)
        raw_data = raw.transpose(2, 0, 1)
        raw_data = np.expand_dims(raw_data, axis=0)
        raw_image = raw_data/(BIT24 - 1)
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
        return rgb_ldr_img

    def load_anno(name):
        a = []
        fn = name
        fn = f"L:/Pycharm/RAOD/datasets/anno/{fn[30:]}"
        fn = f"{fn[:-4]}.json"
        po = []
        i = 0
        with open(fn) as f:
            data = json.load(f)
        for obj in data['shapes']:
            i = i + 1
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
            bo.append((input_size * xmin / 2880 + input_size * xmax / 2880) / 2)
            bo.append((input_size * ymin / 1856 + input_size * ymax / 1856) / 2)
            bo.append(input_size * xmax / 2880 - input_size * xmin / 2880)
            bo.append(input_size * ymax / 1856 - input_size * ymin / 1856)
            bo = np.array(bo)
            po.append(bo)
        for i in range(len(po), 70):
            bo = np.array([0, 0, 0, 0, 0])
            po.append(bo)
        po = np.array(po)
        a.append(po)
        a = np.array(a)
        return Tensor(a)
    path = 'C:/Users/luzhiyi/Desktop/train/day-02000.raw'
    img = load_raw(path)
    img2 = cv2.resize(img, (input_size, input_size))
    img = cv2.resize(img, (input_size, input_size))
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.float()
    img = img.cuda()
    label = load_anno(path)
    label = label.cuda()
    result = model(img, label)
    print(result["total_loss"])
    torch.cuda.empty_cache()
    model.eval()
    result = model(img)
    result = postprocess(result,5)
    print(result)