import numpy as np
import random
import cv2
from .data_augment import get_affine_matrix,apply_affine_to_bboxes

class Dataset:
    def __init__(self, images, labels, input_dim, mosaic_prob=0.5, mixup_prob=0.5, enable_mosaic=True, enable_mixup=True):
        self.images = images
        self.labels = labels
        self.input_dim = input_dim
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.enable_mosaic = enable_mosaic
        self.enable_mixup = enable_mixup
        self.degrees = 10
        self.translate = 0.1
        self.scale = 0.5
        self.shear = 2

    def __len__(self):
        return len(self.images)

    def pull_item(self, index):
        img = self.images[index]
        _labels = self.labels[index]
        img_info = img.shape
        img_id = index
        return img, _labels, img_info, img_id

    def preproc(self, img, labels, input_dim):
        a = img.shape[0]
        img = cv2.resize(img, input_dim)
        try:
            labels[:,1:5] = labels[:,1:5]*(input_dim[0]/a)
        except:
            pass
        return img, labels

    def mixup(self, img, labels, input_dim):
        index = random.randint(0, len(self.images) - 1)
        img2, labels2, _, _ = self.pull_item(index)
        img2 = cv2.resize(img2, input_dim)
        alpha = 0.5
        img = (alpha * img + (1 - alpha) * img2).astype(np.uint8)
        if labels.shape[0] == 0:
            labels = labels2
        elif labels2.shape[0] == 0:
            labels = labels
        else:
            labels = np.vstack((labels, labels2))
        return img, labels


    def random_affine(self,
            img,
            targets,
            target_size,
            degrees=10,
            translate=0.1,
            scales=0.1,
            shear=10,
    ):
        M, scale = get_affine_matrix(target_size, degrees, translate, scales, shear)

        img = cv2.warpAffine(img, M, dsize=target_size, borderValue=(0.5, 0.5, 0.5))

        # Transform label coordinates
        if len(targets) > 0:
            targets = apply_affine_to_bboxes(targets, target_size, M, scale)

        return img, targets

    def get_mosaic_coordinate(self,mosaic_img,mosaic_index, xc, yc, w, h, input_h, input_w):
        if mosaic_index == 0:
            x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
            small_coord = w - (x2 - x1), h - (y2 - y1), w, h
        # index1 to top right part of image
        elif mosaic_index == 1:
            x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, input_w * 2), yc
            small_coord = 0, h - (y2 - y1), min(w, x2 - x1), h
        # index2 to bottom left part of image
        elif mosaic_index == 2:
            x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(input_h * 2, yc + h)
            small_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)
        # index2 to bottom right part of image
        elif mosaic_index == 3:
            x1, y1, x2, y2 = xc, yc, min(xc + w, input_w * 2), min(input_h * 2, yc + h)  # noqa
            small_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)

        large_coord = x1,y1,x2,y2
        return large_coord, small_coord


    def __getitem__(self, idx):
        if self.enable_mosaic and random.random() < self.mosaic_prob:
            mosaic_labels = []
            input_dim = self.input_dim
            input_h, input_w = input_dim[0], input_dim[1]

            yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
            xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))
            yc = int(input_h)
            xc = int(input_w)
            indices = [idx] + [random.randint(0, len(self) - 1) for _ in range(3)]


            for i_mosaic, index in enumerate(indices):
                img, _labels, _, img_id = self.pull_item(index)
                h0, w0 = img.shape[:2]
                scale = min(1. * input_h / h0, 1. * input_w / w0)
                scale = scale
                img = cv2.resize(img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR)
                (h, w, c) = img.shape[:3]
                if i_mosaic == 0:
                    mosaic_img = np.full((input_h * 2, input_w * 2, c), 0.5, dtype=np.float32)

                (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = self.get_mosaic_coordinate(
                    mosaic_img, i_mosaic, xc, yc, w, h, input_h,input_w
                )

                mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2, s_x1:s_x2]
                padw, padh = l_x1 - s_x1, l_y1 - s_y1

                labels = _labels.copy()
                if _labels.size > 0 and _labels.shape != 0:
                    labels[:, 1] = scale * _labels[:, 1] + padw
                    labels[:, 2] = scale * _labels[:, 2] + padh
                    labels[:, 3] = scale * _labels[:, 3] + padw
                    labels[:, 4] = scale * _labels[:, 4] + padh
                    mosaic_labels.append(labels)

            if len(mosaic_labels):
                mosaic_labels = np.concatenate(mosaic_labels, 0)
                np.clip(mosaic_labels[:, 1], 0, 2 * input_w, out=mosaic_labels[:, 1])
                np.clip(mosaic_labels[:, 2], 0, 2 * input_h, out=mosaic_labels[:, 2])
                np.clip(mosaic_labels[:, 3], 0, 2 * input_w, out=mosaic_labels[:, 3])
                np.clip(mosaic_labels[:, 4], 0, 2 * input_h, out=mosaic_labels[:, 4])

            '''mosaic_img, mosaic_labels = self.random_affine(
                mosaic_img,
                mosaic_labels,
                target_size=(input_w, input_h),
                degrees=self.degrees,
                translate=self.translate,
                scales=self.scale,
                shear=self.shear,
            )'''
            '''if (
                self.enable_mixup
                and not len(mosaic_labels) == 0
                and random.random() < self.mixup_prob
            ):
                mosaic_img, mosaic_labels = self.mixup(mosaic_img, mosaic_labels, self.input_dim)'''
            mix_img, padded_labels = self.preproc(mosaic_img, mosaic_labels, self.input_dim)
            img_info = (mix_img.shape[1], mix_img.shape[0])
            return mix_img, padded_labels, img_info, img_id

        else:
            img, label, img_info, img_id = self.pull_item(idx)
            img, label = self.preproc(img, label, self.input_dim)
            return img, label, img_info, img_id

def augment_batch(images, labels, input_dim):
    images = images.transpose(0,2,3,1)
    dataset = Dataset(images, labels, input_dim)
    augmented_images = []
    augmented_labels = []

    for i in range(images.shape[0]):
        img, label, _, _ = dataset[i]
        label = np.array(label)
        img = img.transpose((2, 0, 1))
        augmented_images.append(img)
        augmented_labels.append(label)

    return augmented_images, augmented_labels



