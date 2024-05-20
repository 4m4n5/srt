import os
import numpy as np
import torch
from torch.utils.data import Dataset

class AirplaneDataset(Dataset):
    def __init__(self, path, split='train'):
        self.path = path

        self.render_kwargs = {
            'min_dist': 2.,
            'max_dist': 4.}

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
        # import pdb; pdb.set_trace()
        idx1, idx2 = np.random.randint(100), np.random.randint(100)
        folder = os.path.join(self.path,'img/')+self.objects[idx]+'/'
        ray_folder = os.path.join(self.path,'rays/')+self.objects[idx]+'/'
        
        img1, img2 = np.load(folder+'lores'+str(idx1)+'.npy'), np.load(folder+'lores'+str(idx2)+'.npy')
        img1, img2 = img1.astype(np.float32).transpose(2,0,1)/255, img2.astype(np.float32).transpose(2,0,1)/255
        R1, R2 = np.load(folder+str(idx1)+'.npy'), np.load(folder+str(idx2)+'.npy')
        R1, R2 = R1.reshape(-1), R2.reshape(-1)
        
        img1 = np.expand_dims(img1,0)
        
        img2 = img2.transpose((1,2,0)).reshape(-1,3)

        result = {
            'input_images': img1,
            'input_camera_pos': 0,
            'input_rays': R1,              #Now, just (9,)
            'target_pixels': img2,
            'target_camera_pos': 0,
            'target_rays': R2,             #Now, just (9,)
            'sceneid': idx,
        }

        return result