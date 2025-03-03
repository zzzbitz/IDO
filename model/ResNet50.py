import torch
import torch.nn as nn
import timm
import sys
import os
# 将utils文件夹添加到sys.path 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
from config import _C as cfg
import torch.nn.functional as F

class ResNet50(nn.Module):
    def __init__(self, cfg): 
        super(ResNet50, self).__init__() 
        self.stage = cfg.stage
        print(f'Loading ResNet50, pretrained:{cfg.pretrained}, stage:{cfg.stage}, num_class:{cfg.num_class}')
        self.model = timm.create_model('resnet50', pretrained=cfg.pretrained, features_only=True) 
        self.fc = nn.Linear(self.model.feature_info[-1]['num_chs'], cfg.num_class)
        if cfg.stage == 1:
            self.freeze(cfg.frozen)
            print(f'In stage 1, whether freeze the feature extractor:{cfg.frozen}')
        elif cfg.stage == 2:
            print(f'In stage 2, fully fine tuned the model')
            self.freeze(False)
        else:
            raise ValueError("please check the value of stage, it should be 1 or 2" )
        
    def forward(self, x):
        features_list = self.model(x)  
        features = features_list[-1]   
        

        features = F.adaptive_avg_pool2d(features, (1, 1))
        
        
        features = features.view(features.size(0), -1)
        
      
        logits = self.fc(features)
        
        return features, logits

    
    def freeze(self, freeze=False):
        for param in self.model.parameters():
            param.requires_grad = not freeze

if __name__ == '__main__':
    cfg.defrost()
    cfg.merge_from_file("./config/Stage_2/cifar100.yaml")
    model = ResNet50(cfg)
    print(model)
