import torch
import torch.nn as nn
import timm
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
from config import _C as cfg

class ViTBase16(nn.Module):
    def __init__(self, cfg):
        super(ViTBase16, self).__init__()
        self.stage = cfg.stage
        print(f'Loading ViT Base 16, pretrained:{cfg.pretrained}, stage:{cfg.stage}, num_class:{cfg.num_class}')
        
        self.model = timm.create_model('vit_base_patch16_224', pretrained=cfg.pretrained, num_classes=0)
        
        self.fc = nn.Linear(768, cfg.num_class)
        if cfg.stage == 1:
            self.freeze(cfg.frozen)
            print(f'In stage 1, whether freeze the feature extractor:{cfg.frozen}')
        elif cfg.stage == 2:
            print(f'In stage 2, fully fine-tuned the model')
            self.freeze(False)
        else:
            raise ValueError("Please check the value of stage, it should be 1 or 2")
    
    def forward(self, x):
        features = self.model(x)  
        logits = self.fc(features)  
        return features, logits
    
    def freeze(self, freeze=False):
        for param in self.model.parameters():
            param.requires_grad = not freeze

if __name__ == '__main__':
    cfg.defrost()
    cfg.merge_from_file("./config/Stage_2/cifar100.yaml")
    model = ViTBase16(cfg)
    print(model)
