import torch
import torch.nn as nn
import timm
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
from config import _C as cfg
import torch.nn.functional as F

class ConvNeXtB(nn.Module):
    def __init__(self, cfg):
        super(ConvNeXtB, self).__init__()
        self.stage = cfg.stage
        print(f'Loading ConvNeXt B, pretrained:{cfg.pretrained}, stage:{cfg.stage}, num_class:{cfg.num_class}')
        self.model = timm.create_model('convnext_base', pretrained=cfg.pretrained, features_only=True)
        self.fc = nn.Linear(self.model.feature_info[-1]['num_chs'], cfg.num_class)
        if cfg.stage == 1:
            self.freeze(cfg.frozen)
            print(f'In stage 1, whether freeze the feature extractor:{cfg.frozen}')
        elif cfg.stage == 2:
            print(f'In stage 2, fully fine-tuned the model')
            self.freeze(False)
        else:
            raise ValueError("Please check the value of stage, it should be 1 or 2")

    def forward(self, x):
        features = self.model(x)[-1]
        # 全局平均池化，将特征图转换为 [batch_size, 2048, 1, 1]
        features = F.adaptive_avg_pool2d(features, (1, 1))
        
        # 展平成 [batch_size, 2048]
        features = features.view(features.size(0), -1)
        logits = self.fc(features)
        return features, logits

    def freeze(self, freeze=False):
        for param in self.model.parameters():
            param.requires_grad = not freeze

if __name__ == '__main__':
    cfg.defrost()
    cfg.merge_from_file("/mnt/lustre/zhonghuaping.p/zhangkuan/KDD2025/OtherPaperCode/IDO/config/Stage_2/cifar100.yaml")
    model = ConvNeXtB(cfg)
    print(model)