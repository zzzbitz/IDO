import torch
import torch.nn as nn
import timm
import sys
import os
# 将utils文件夹添加到sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
from config import _C as cfg

class ViTBase16(nn.Module):
    def __init__(self, cfg):
        super(ViTBase16, self).__init__()
        self.stage = cfg.stage
        print(f'Loading ViT Base 16, pretrained:{cfg.pretrained}, stage:{cfg.stage}, num_class:{cfg.num_class}')
        # 设置 num_classes=0 以去除原始的分类头
        self.model = timm.create_model('vit_base_patch16_224', pretrained=cfg.pretrained, num_classes=0)
        # 添加新的全连接层
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
        features = self.model(x)  # 输出特征
        logits = self.fc(features)  # 新的分类层
        return features, logits
    
    def freeze(self, freeze=False):
        for param in self.model.parameters():
            param.requires_grad = not freeze

if __name__ == '__main__':
    cfg.defrost()
    cfg.merge_from_file("/mnt/lustre/zhonghuaping.p/zhangkuan/KDD2025/OtherPaperCode/IDO/config/Stage_2/cifar100.yaml")
    model = ViTBase16(cfg)
    print(model)