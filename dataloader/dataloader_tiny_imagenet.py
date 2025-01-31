import os
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch
import json
import random
from .randaugment import TransformFixMatchLarge
import sys

def build_loader(dataset="tinyimagenet", data_path="", resolution=224, noise_mode="sym", noise_ratio=0.2, num_workers=1, batch_size=64, max_samples=10000):
    print(f'Information: dataset-{dataset}, noise_mode-{noise_mode}, noise_ratio-{noise_ratio}, max_samples-{max_samples}')
    sys.stdout.flush()
    
    train_set = ImageDataset(train=True, root=data_path, resolution=resolution, 
                            noise_mode=noise_mode, noise_ratio=noise_ratio, 
                            stage=2, max_samples=max_samples)
    train_loader = torch.utils.data.DataLoader(train_set, num_workers=num_workers, 
                                             shuffle=True, batch_size=batch_size)

    eval_set = ImageDataset(train=True, root=data_path, resolution=resolution,
                           noise_mode=noise_mode, noise_ratio=noise_ratio,
                           stage=2, max_samples=max_samples)
    eval_loader = torch.utils.data.DataLoader(eval_set, num_workers=num_workers,
                                            shuffle=False, batch_size=batch_size)

    test_set = ImageDataset(train=False, root=data_path, resolution=resolution,
                           noise_mode=noise_mode, noise_ratio=noise_ratio,
                           stage=2, max_samples=max_samples)
    test_loader = torch.utils.data.DataLoader(test_set, num_workers=num_workers,
                                            shuffle=True, batch_size=batch_size)

    return train_loader, eval_loader, test_loader

class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, 
                 train: bool,
                 root: str,
                 resolution: int,
                 noise_mode='sym',
                 noise_ratio=0.0,
                 stage=2,
                 max_samples=None  # 新添加的参数
                 ):
        """ basic information """
        self.root = root
        self.resolution = resolution
        self.train = train
        self.stage = stage
        self.max_samples = max_samples
        
        os.makedirs(os.path.join(root, 'noise_file'), exist_ok=True)

        """ declare data augmentation """
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        if self.train:
            if self.stage == 1:
                self.transforms = transforms.Compose([
                            transforms.Resize((256, 256), Image.BILINEAR),
                            transforms.RandomCrop((resolution, resolution)),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std)
                    ])
            elif self.stage == 2:
                self.transforms = TransformFixMatchLarge(mean, std)
            else:
                raise ValueError("please check the value of stage, it should be 1 or 2")
        else:
            self.transforms = transforms.Compose([
                        transforms.Resize((resolution, resolution), Image.BILINEAR),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std)
                ])

        """ read all data information """
        if self.train:
            data, label = self.getDataInfo(os.path.join(root, "train"))
        else:
            data, label = self.getDataInfo(os.path.join(root, "test"))

        # 限制数据量
        if self.max_samples is not None and self.max_samples < len(data):
            print(f"Limiting dataset size from {len(data)} to {self.max_samples}")
            # 使用相同的随机种子确保每次获取相同的子集
            random.seed(42)
            indices = random.sample(range(len(data)), self.max_samples)
            data = [data[i] for i in indices]
            label = [label[i] for i in indices]

        if self.train:
            data_num = len(data)
            noise_file = os.path.join(self.root, 'noise_file', noise_mode + '_' + str(noise_ratio))

            if os.path.exists(noise_file):
                noise_label = json.load(open(noise_file,"r"))
                if self.max_samples is not None and self.max_samples < len(noise_label):
                    noise_label = [noise_label[i] for i in indices]
            elif noise_mode == 'idn':
                img = []
                for path in data:
                    image = Image.open(path).convert('RGB')
                    image_array = np.array(image)
                    img.append(image_array.flatten())    
                img = np.array(img)
                img = torch.from_numpy(img).float()
                print(img[0])
                targets = torch.from_numpy(np.array(label))
                dataset = zip(img, targets)
                noise_label = self.get_instance_noisy_label(noise_ratio, dataset, targets, self.num_class, feature_size=3*64*64)
                print("save noisy labels to %s ..."%noise_file)        
                json.dump(noise_label,open(noise_file,"w"))   
            else:
                noise_label = []
                idx = list(range(data_num))
                random.shuffle(idx)
                num_noise = int(noise_ratio * data_num)            
                noise_idx = idx[:num_noise]
                asym_transition = {}
                for i in range(self.num_class - 1):
                    asym_transition[i] = i + 1
                asym_transition[self.num_class - 1] = 0
                noise_class = list(range(self.num_class))
                random.shuffle(noise_class)
                noise_class = noise_class[:int(self.num_class / 2)]

                for i in range(data_num):
                    if i in noise_idx:
                        if noise_mode=='sym':
                            noiselabel = random.randint(0, self.num_class - 1)
                            noise_label.append(noiselabel)
                        elif noise_mode=='asym':   
                            noiselabel = asym_transition[label[i]]
                            noise_label.append(noiselabel) 
                        elif noise_mode == 'str':
                            if label[i] in noise_class:
                                # noiselabel = random.randint(0, self.num_class - 1)
                                noiselabel = asym_transition[label[i]]
                            else:
                                noiselabel = label[i]
                            noise_label.append(noiselabel)                  
                    else:    
                        noise_label.append(label[i])   
                print("save noisy labels to %s ..."%noise_file)        
                json.dump(noise_label, open(noise_file,"w"))  

            self.data = data
            self.label = noise_label
            self.noise_idx = torch.zeros(data_num, dtype=torch.bool)
            for i in range(data_num):
                if label[i] != noise_label[i]:
                    self.noise_idx[i] = 1
            self.noise_label = noise_label
            self.clean_label = label
        else:
            self.data = data
            self.label = label

    def get_instance_noisy_label(self, n, dataset, labels, num_classes, feature_size, norm_std=0.1, seed=1): 
        # n -> noise_rate 
        # dataset -> mnist, cifar10 # not train_loader
        # labels -> labels (targets)
        # label_num -> class number
        # feature_size -> the size of input images (e.g. 28*28)
        # norm_std -> default 0.1
        # seed -> random_seed 
        from math import inf
        from scipy import stats

        print("building dataset...")
        label_num = num_classes
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        torch.cuda.manual_seed(int(seed))

        P = []
        flip_distribution = stats.truncnorm((0 - n) / norm_std, (1 - n) / norm_std, loc=n, scale=norm_std)
        flip_rate = flip_distribution.rvs(labels.shape[0])

        if isinstance(labels, list):
            labels = torch.FloatTensor(labels)
        labels = labels.cuda()

        W = np.random.randn(label_num, feature_size, label_num)


        W = torch.FloatTensor(W).cuda()
        for i, (x, y) in enumerate(dataset):
            # 1*m *  m*10 = 1*10
            x = x.cuda()
            A = x.view(1, -1).mm(W[y]).squeeze(0)
            A[y] = -inf
            A = flip_rate[i] * torch.nn.functional.softmax(A, dim=0)
            A[y] += 1 - flip_rate[i]
            P.append(A)
        P = torch.stack(P, 0).cpu().numpy()
        l = [i for i in range(label_num)]
        new_label = [int(np.random.choice(l, p=P[i])) for i in range(labels.shape[0])]
        record = [[0 for _ in range(label_num)] for i in range(label_num)]

        for a, b in zip(labels, new_label):
            a, b = int(a), int(b)
            record[a][b] += 1


        pidx = np.random.choice(range(P.shape[0]), 1000)
        cnt = 0
        for i in range(1000):
            if labels[pidx[i]] == 0:
                a = P[pidx[i], :]
                cnt += 1
            if cnt >= 10:
                break

        return new_label  

    def __len__(self):
        return len(self.data)

    def getDataInfo(self, root):
        data = []
        label = []
        folders = os.listdir(root)
        folders.sort() # sort by alphabet
        self.num_class = len(folders)
        for class_id, folder in enumerate(folders):
            files = os.listdir(os.path.join(root, folder, "images"))
            for file in files:
                data_path = os.path.join(root, folder, "images", file)
                data.append(data_path)
                # data.append(np.expand_dims(cv2.imread(data_path)[:, :, ::-1], 0))
                label.append(class_id)
        # print(data)
        # data = np.concatenate(data)
        return data, label

    def __getitem__(self, index):
        # get data information
        img = cv2.imread(self.data[index])
        img = img[:, :, ::-1] # BGR to RGB.
        img = Image.fromarray(img)
        img = self.transforms(img)
        
        label = self.label[index]
        
        if self.train:
            return img, label, index
        
        # return img, sub_imgs, label, sub_boundarys
        return img, label