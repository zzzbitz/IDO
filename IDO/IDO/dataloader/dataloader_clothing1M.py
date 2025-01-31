from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import torch
from .randaugment import TransformFixMatchLarge
import sys

class clothing_dataset(Dataset): 
    def __init__(self, root, transform, mode, num_samples=0, num_class=14): 
        
        self.root = root
        self.transform = transform
        self.mode = mode
        self.train_labels = {}
        self.test_labels = {} 
        self.num_class = num_class         
        
        with open('%s/annotations/noisy_label_kv.txt'%self.root,'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()           
                img_path = '%s/'%self.root+entry[0]
                self.train_labels[img_path] = int(entry[1])                         
        with open('%s/annotations/clean_label_kv.txt'%self.root,'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()           
                img_path = '%s/'%self.root+entry[0]
                self.test_labels[img_path] = int(entry[1])   

        if mode == 'test':                                       
            self.test_imgs = []
            with open('%s/annotations/clean_test_key_list.txt'%self.root,'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/'%self.root+l
                    self.test_imgs.append(img_path)    
        else:
            train_imgs=[]
            with open('%s/annotations/noisy_train_key_list.txt'%self.root,'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/'%self.root+l
                    train_imgs.append(img_path)                                
            random.shuffle(train_imgs)
            class_num = torch.zeros(num_class)
            self.train_imgs = []
            for impath in train_imgs:
                label = self.train_labels[impath] 
                if class_num[label]<(num_samples/14) and len(self.train_imgs)<num_samples:
                    self.train_imgs.append(impath)
                    class_num[label]+=1
            random.shuffle(self.train_imgs)              
                    
    def __getitem__(self, index):  
        if self.mode=='train':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]     
            image = Image.open(img_path).convert('RGB')   
            img = self.transform(image)
            return img, target, index      
        elif self.mode=='eval':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]     
            image = Image.open(img_path).convert('RGB')   
            img = self.transform(image)
            return img, target, index     
        elif self.mode=='test':
            img_path = self.test_imgs[index]
            target = self.test_labels[img_path]     
            image = Image.open(img_path).convert('RGB')   
            img = self.transform(image) 
            return img, target
        
    def __len__(self):
        if self.mode=='test':
            return len(self.test_imgs)
        else:
            return len(self.train_imgs)            

def build_loader(cfg):      
    if cfg.stage == 1:    
        transform_train = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),                
                transforms.Normalize((0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)),                     
            ]) 
    elif cfg.stage == 2:
        transform_train =  TransformFixMatchLarge(
            (0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214))
    else:
        raise ValueError("please check the value of stage, it should be 1 or 2")
    transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)),
        ])   
               
    print(f'Infomation: dataset-{cfg.dataset}, noise_mode-{cfg.noise_mode}, noise_ratio-{cfg.noise_ratio}')
    sys.stdout.flush()
    train_dataset = clothing_dataset(cfg.data_path, transform=transform_train, mode='train', num_samples=1000 * cfg.batch_size)
    train_loader = DataLoader(dataset=train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)  
    eval_dataset = clothing_dataset(cfg.data_path, transform=transform_test, mode='eval', num_samples=1000 * cfg.batch_size)
    eval_loader = DataLoader(dataset=eval_dataset, batch_size=cfg.batch_size * 2, shuffle=False, num_workers=cfg.num_workers)  
    test_dataset = clothing_dataset(cfg.data_path, transform=transform_test, mode='test')
    test_loader = DataLoader(dataset=test_dataset, batch_size=cfg.batch_size,shuffle=False,num_workers=cfg.num_workers)      

    return train_loader, eval_loader, test_loader