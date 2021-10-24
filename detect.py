

import os
import cv2
import time
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from src.PFLD import PFLDInference

os.environ['CUDA_VISIBLE_DEVICES']="0"


img_root = '/root/PangjPro/CLASS_Traffic_Signal/Dataset_Traffic_Light/val'
sav_dir = '/root/PangjPro/CLASS_Traffic_Signal/results'

model_path = '/root/PangjPro/CLASS_Traffic_Signal/best.pth'
image_size = 224



###  预处理
def image_pro(img):

    H,W,C = img.shape
    rat = min(image_size/H, image_size/W)
    new_unpad = (int(round(W * rat)), int(round(H * rat)))
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)            

    H,W,C = img.shape
    top = (image_size - H)//2
    bom = image_size - H - top
    lft = (image_size - W)//2
    rgt = image_size - W - lft
    img = cv2.copyMakeBorder(img, top, bom, lft, rgt, cv2.BORDER_CONSTANT, value=(0,0,0)) 

    img = np.transpose(img,(2,0,1))/255.0
    img = img.astype(np.float32)
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)

    return img





# build model
model = PFLDInference(num_classes=4)

if torch.cuda.is_available():
    model = model.cuda()

model.load_state_dict(torch.load(model_path))


model.eval()

pic_num = 0
for img_cls in os.listdir(img_root):
    img_dir = img_root + '/' + str(img_cls)
    for imgfile in os.listdir(img_dir):
        img_path = img_dir + '/' + imgfile


        try:

            imgcv = cv2.imread(img_path)
            H,W,C = imgcv.shape

            if min(H,W)<25:
                continue

            img = image_pro(imgcv.copy())

            image = img.cuda()
            out = model(image)
            out = out.cpu().data.numpy()[0]




            ###后处理
            """
            dic = np.argmax(out)
            if out[dic]<0.3:
                continue
            sav_path = sav_dir + '/' + str(dic) + '/' +  '2_'+imgfile
            cv2.imwrite(sav_path, imgcv)
            
            """
            sav_path = sav_dir + '/' +   imgfile[:-4]
            for i in range(4):
                sav_path += '_' + str(int(100*float(out[i])))
            
            
            cv2.imwrite(sav_path+'.jpg', imgcv)
            



            pic_num += 1
            print(pic_num)




        except UnicodeEncodeError as e:
            print ("UnicodeEncodeError Details : " + str(e))
            pass








