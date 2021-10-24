
import os
import cv2
import time
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from src.dataset import CLASSDataset
from src.PFLD import PFLDInference
from src.loss import CLASSNetLoss as loss_f

os.environ['CUDA_VISIBLE_DEVICES']="0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config = {'train_root':'./Pascal_voc2007/train' , 
          'val_root':'./Pascal_voc2007/val' , 
          'permodel':'' , 
          'bacth_size': 4 ,
          'base_lr':0.0001}









if torch.cuda.is_available():
    torch.cuda.manual_seed(123)
else:
    torch.manual_seed(123)


# build Dataset
train_dataset = CLASSDataset(config['train_root'], train = True, image_size=192)
dataloader_train  =  DataLoader(train_dataset,
                                batch_size = config['bacth_size'],
                                shuffle = True,
                                num_workers = 4,
                                drop_last=True)

val_dataset = CLASSDataset(config['val_root'], train = False, image_size=192)
dataloader_val  =  DataLoader(val_dataset,
                                batch_size = config['bacth_size'],
                                shuffle = True,
                                num_workers = 4,
                                drop_last=True)

print(train_dataset.class_dict)

print(train_dataset.class_wgt)


# build model
model = PFLDInference(num_classes=len(train_dataset.class_dict))
model = model.to(device)

# build optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config['base_lr'], betas=(0.9, 0.99))
criterion = loss_f(class_wgt = train_dataset.class_wgt)




def train(epoch, model):
    model.train()

    if epoch > 15:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.00001
    if epoch > 50:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.000001

    loss_train = 0
    for i, (image, label) in enumerate(dataloader_train):
        if torch.cuda.is_available():
            image = image.to(device)
            label = label.to(device)

        out = model(image)

        loss = criterion(out, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i%10 == 0:
            print(epoch, i, loss.item() , optimizer.param_groups[0]['lr'])
        loss_train += loss.item()

        #if i > 100:
        #    break

    return loss_train/len(dataloader_train)







def evaluation(epoch, model):

    model.eval()

    num_all = [0]*len(train_dataset.class_dict)
    num_tp  = [0]*len(train_dataset.class_dict)

    for i, (image, label) in enumerate(dataloader_val):
        if torch.cuda.is_available():
            image = image.to(device)


        out = model(image)
        if torch.cuda.is_available():
            out = out.cpu()
        out = out.data.numpy()

        lbl = label.data.numpy()

        B,C = out.shape
        for b in range(B):
            tgt = np.argmax(lbl[b,:])
            prp = np.argmax(out[b,:])

            if tgt == prp:
                num_tp[tgt] += 1
            num_all[tgt] += 1

        if i%20 == 0:
            print(i)

        #if i > 100:
        #    break

    all_ap = np.sum(num_tp)/np.sum(num_all)
    men_ap = []
    for c in range(C):
        one_ap = num_tp[c]/num_all[c]
        men_ap.append(one_ap)
        print(one_ap)
    men_ap = np.mean(men_ap)

    return men_ap , all_ap




    



start_time = time.time()        
epochs = 100
best_model_ap = 0.0
start_epoch = 0
for epoch in range(start_epoch, epochs, 1): 
    loss_train = train(epoch, model)

    men_ap , all_ap = evaluation(epoch, model)

    print(epoch , loss_train , men_ap , all_ap)
    if men_ap>best_model_ap:
        torch.save(model.state_dict(), 'best.pth')
        best_model_ap = men_ap
    torch.save(model.state_dict(), 'last.pth')

    loss_txt = open('train_loss.txt', 'a+')
    loss_txt.read()
    loss_txt.write(str(epoch) + ' ')
    loss_txt.write(str(round(loss_train, 4)) + ' ')
    loss_txt.write(str(round(men_ap, 4)) + ' ')
    loss_txt.write(str(round(all_ap, 4)) + ' ')
    loss_txt.write('\n')
    


    

