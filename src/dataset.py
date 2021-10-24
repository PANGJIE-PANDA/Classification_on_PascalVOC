#encoding:utf-8
#
#created by xiongzihua
#
import os
import sys
import random
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader

import cv2

class CLASSDataset(data.Dataset):
    def __init__(self, root, train, image_size=224):
        print('data init')
        self.image_size = image_size
        self.root = root
        self.train = train

        self.class_dict = ['chair', 'diningtable', 'cat', 'boat', 'sofa', 'bird', 'bottle', 'person', 
                           'tvmonitor', 'car', 'bus', 'bicycle', 'sheep', 'motorbike', 'pottedplant', 
                           'dog', 'train', 'horse', 'cow', 'aeroplane']


        self.img_dict = []
        self.lbl_dict = []
        self.class_wgt = []
        for cls_id in range(len(self.class_dict)):
            img_cls = self.class_dict[cls_id]
            img_dir = self.root + '/' + img_cls
            for imgfile in os.listdir(img_dir):
                img_path = img_dir + '/' + imgfile
                self.img_dict.append(img_path)
                self.lbl_dict.append(self.class_dict.index(img_cls))

            self.class_wgt.append(len(os.listdir(img_dir)))

        self.num_samples = len(self.img_dict)



    def __getitem__(self,idx):
        img_path = self.img_dict[idx]
        lbl_id = self.lbl_dict[idx]

        img = cv2.imread(img_path)
        
        if self.train:
            img = self.random_bright(img)
            img = self.randomBlur(img)
            img = self.randomflipx(img)
            img = self.randomflipy(img)
            img = self.randomcut(img)
            img = self.randomsize(img)
            img = self.randomrot(img)

        img_tensor = self.img_to_tensor(img)
        
        lbl = np.zeros((len(self.class_dict),), dtype=np.int)
        lbl[lbl_id] = 1
    
        return img_tensor, lbl
        


    def __len__(self):
        return self.num_samples

    def img_to_tensor(self, bgr):
        H,W,C = bgr.shape
        rat = min(self.image_size/H, self.image_size/W)
        new_unpad = (int(round(W * rat)), int(round(H * rat)))
        bgr = cv2.resize(bgr, new_unpad, interpolation=cv2.INTER_LINEAR)   
        H,W,C = bgr.shape
        top = (self.image_size - H)//2
        bom = self.image_size - H - top
        lft = (self.image_size - W)//2
        rgt = self.image_size - W - lft
        bgr = cv2.copyMakeBorder(bgr, top, bom, lft, rgt, cv2.BORDER_CONSTANT, value=(0,0,0)) 

        bgr = np.transpose(bgr,(2,0,1))/255.0
        bgr = bgr.astype(np.float32)
        bgr = torch.from_numpy(bgr)

        return bgr




    def randomBlur(self, bgr):
        if random.random()<0.5:
            bgr = cv2.blur(bgr,(5,5))
        return bgr

    def randomflipx(self, bgr):
        if random.random()<0.5:
            bgr = cv2.flip(bgr, 1) 
        return bgr  

    def randomflipy(self, bgr):
        if random.random()<0.5:
            bgr = cv2.flip(bgr, 0) 
        return bgr  

    def random_bright(self, im, delta=16):
        alpha = random.random()
        if alpha > 0.3:
            im = im * alpha + random.randrange(-delta,delta)
            im = im.clip(min=0,max=255).astype(np.uint8)
        return im

    def randomcut(self, bgr):

        H,W,_ = bgr.shape
        xs = 0.2
        if random.random()<0.5:
            xc = random.randint(-int(xs*W), int(xs*W))
            yc = random.randint(-int(xs*H), int(xs*H))
            if xc<0:
                bgr = bgr[: , :xc , :]
            else:
                bgr = bgr[: , xc: , :]
            if yc<0:
                bgr = bgr[:yc , : , :]
            else:
                bgr = bgr[yc: , : , :]

        return bgr  



    def randomsize(self, bgr):

        H,W,_ = bgr.shape
        xr = (random.random()-0.5)/3 + 1
        yr = (random.random()-0.5)/3 + 1

        bgr = cv2.resize(bgr, (int(xr*W) , int(yr*H)), interpolation=cv2.INTER_LINEAR)

        return bgr  

    def randomrot(self, bgr):

        if random.random()<0.5:
            H,W,C = bgr.shape
            center = (W // 2, H // 2)
            rt = random.randint(0,90)
            M = cv2.getRotationMatrix2D(center, rt, 1.0) #15
            bgr = cv2.warpAffine(bgr, M, (W, H)) #16

        return bgr  



if __name__ == '__main__':
    train_dataset = CLASSDataset('../Pascal_voc2007/train', train = True, image_size=224)

    dataloader_train  =  DataLoader(train_dataset, batch_size = 4, shuffle = True, num_workers = 0, drop_last=True)

    for i, (image, label) in enumerate(dataloader_train):

        print(image.shape , label.shape)
