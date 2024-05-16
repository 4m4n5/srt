import os
import numpy as np
import torch
from torch.utils.data import Dataset

class AirplaneDataset(Dataset):
    def __init__(self, path, split='train'):
        self.path = path
        if split=='train':
            objects = np.load(os.path.join(path,'train_lst.npy'))
            self.objects = objects[:int(len(objects)/8*7)]
        elif split=='val':
            objects = np.load(os.path.join(path,'train_lst.npy'))
            self.objects = objects[int(len(objects)/8*7):]
        elif split=='test':
            self.objects = np.load(os.path.join(path,'test_lst.npy'))
                
    def __len__(self):
        return len(self.objects)
    
    def __getitem__(self, idx):
        idx1, idx2 = np.random.randint(100), np.random.randint(100)
        folder = os.path.join(self.path,'img/')+self.objects[idx]+'/'
        img1, img2 = np.load(folder+'lores'+str(idx1)+'.npy'), np.load(folder+'lores'+str(idx2)+'.npy')
        img1, img2 = img1.astype(np.float32).transpose(2,0,1)/255, img2.astype(np.float32).transpose(2,0,1)/255
        p1, p2 = np.load(folder+'pos'+str(idx1)+'.npy'), np.load(folder+'pos'+str(idx2)+'.npy')
        R1, R2 = np.load(folder+'ray'+str(idx1)+'.npy'), np.load(folder+'ray'+str(idx2)+'.npy')
        img1 = np.expand_dims(img1,0)
        p1 = np.expand_dims(p1,0)
        R1 = np.expand_dims(R1,0)
        img2 = img2.transpose((1,2,0)).reshape(-1,3)
        p2 = np.expand_dims(p2,0).repeat(64*64,0)
        R2 = R2.reshape(-1,3)
        return img1, p1, R1, img2, p2, R2, idx
