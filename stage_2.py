import os
import sys
import random
import argparse
import numpy as np
import time
import torch
from utils.config import _C as cfg
from utils.bmm import *
from utils.lnl_methods import *
from model import ViT16B, ResNet50, ConvNeXtB
import timm
import line_profiler

# config setting
parser = argparse.ArgumentParser()
parser.add_argument("--cfg", type=str, default="", help="path to config file")
parser.add_argument("--noise_mode", default=None)
parser.add_argument("--noise_ratio", default=None)
parser.add_argument("--gpuid", default=None)
parser.add_argument("--backbone", default=None)

args = parser.parse_args()
cfg.defrost()
cfg.merge_from_file(args.cfg)
if args.noise_mode is not None:
    cfg.noise_mode = args.noise_mode
if args.noise_ratio is not None:
    cfg.noise_ratio = float(args.noise_ratio)
if args.gpuid is not None:
    cfg.gpuid = int(args.gpuid)
if args.backbone is not None:
    cfg.backbone = args.backbone

def set_seed():
    torch.cuda.set_device(cfg.gpuid)
    seed = cfg.seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed()

# Train
def train(epoch, dataloader):
    model.train()
    num_iter = (len(dataloader.dataset) // dataloader.batch_size) + 1
    correct, total = 0, 0

    for batch_idx, (inputs, targets, index) in enumerate(dataloader):
        # load batch data
        weak_inputs, strong_inputs, targets = inputs[0].cuda(non_blocking=True), inputs[1].cuda(non_blocking=True), targets.cuda()

        # feed-forward
        _, weak_outputs = model(weak_inputs)
        _, strong_outputs = model(strong_inputs)

        # update wrong event
        weak_prediction = F.log_softmax(weak_outputs, dim=1)
        __, predicted = torch.max(weak_prediction.data, 1)
        
        strong_prediction = F.log_softmax(strong_outputs, dim=1)
        __, strong_predicted = torch.max(strong_prediction.data, 1)

        acc_w = (predicted.detach_().cpu().numpy() == targets.detach_().cpu().numpy())
        acc_s = (strong_predicted.detach_().cpu().numpy() == targets.detach_().cpu().numpy())
        index_arr = np.array(index)
        wrong_event[index_arr[~acc_s]] += 1
        wrong_event[index_arr[~acc_w]] += 1

        # get information from bmm(whole dataset bmm)
        prob1, prob2, cdf1, cdf2 = bmm_model.predict(wrong_event[index_arr])

        # calculate loss and update model
        loss = criterion(weak_outputs, strong_outputs, prob1, prob2, cdf1, cdf2, targets)

        loss.backward()
        optimizer.step()
       
        optimizer.zero_grad()
        total += targets.size(0)
        correct += predicted.eq(targets).cpu().sum().item()

        sys.stdout.write('\r')
        sys.stdout.write('Epoch [%3d/%3d] Iter[%3d/%3d]\t loss: %.3f'
                %( epoch, cfg.epochs, batch_idx+1, num_iter, loss.item()))
        sys.stdout.flush()

    print("\n| Train Epoch #%d\t Accuracy: %.2f\n" %(epoch, 100. * correct/total))
    return 100.*correct/total

# Test
def test(epoch, dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()
    acc = 100. * correct / total
    print("\n| Test Epoch #%d\t Accuracy: %.2f\n" %(epoch, acc))
    return acc

# ======== Data ========
if cfg.dataset == "clothing1m":
    from dataloader import dataloader_clothing1M as dataloader
    train_loader, _, test_loader = dataloader.build_loader(cfg)
elif cfg.dataset == "webvision":
    from dataloader import dataloader_webvision as dataloader
    train_loader, _, test_loader = dataloader.build_loader(cfg)
elif cfg.dataset.startswith("cifar"):
    from dataloader import dataloader_cifar as dataloader
    loader = dataloader.cifar_dataloader(cfg.dataset, noise_mode=cfg.noise_mode, noise_ratio=cfg.noise_ratio,\
                                        batch_size=cfg.batch_size, num_workers=cfg.num_workers, root_dir=cfg.data_path, model=cfg.model, stage=cfg.stage, pretrained=cfg.pretrained)
    train_loader = loader.run('train')
    test_loader = loader.run('test')
elif cfg.dataset == "tiny_imagenet":
    print("Loading Tiny-ImageNet...")
    from dataloader import dataloader_tiny_imagenet as dataloader
    train_loader, _, test_loader = dataloader.build_loader(cfg)

num_class = cfg.num_class



# ======== Model ========
if cfg.backbone == 'ViT-B':
    model = ViT16B.ViTBase16(cfg)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, weight_decay=1e-5)
elif cfg.backbone == 'resnet50':
    model = ResNet50.ResNet50(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
elif cfg.backbone == 'convnext-B':
    model = ConvNeXtB.ConvNeXtB(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
else:
    raise ValueError('please check the backbone of the model')

if cfg.pretrained == True:
    model.load_state_dict(torch.load("./base_model/{}_pretrained_{}_{}_{}.pt".format(cfg.backbone, cfg.dataset, cfg.noise_mode, cfg.noise_ratio)))
else:
    model.load_state_dict(torch.load("./base_model/{}_scratch_{}_{}_{}.pt".format(cfg.backbone, cfg.dataset, cfg.noise_mode, cfg.noise_ratio)))
model.cuda()

wrong_event = load_wrong_event(cfg)
criterion = WELoss(num_classes=cfg.num_class)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min = 1e-4)
bmm_model = BetaMixtureModel()
bmm_model.fit(wrong_event, max_iter=20)
print(f'image shape:{train_loader.dataset[0][0][0].shape}')

# check requires_grad
for param in model.parameters(): 
    param.requires_grad = True 
bmm_model.print_beta()
sys.stdout.flush()

best_acc = 0
wrong_event = np.array(wrong_event)

# noise-robust training process
for epoch in range(1, cfg.epochs + 1):
    train_acc = train(epoch, train_loader)
    bmm_model.fit(wrong_event, max_iter=1)
    print(bmm_model.print_beta())
    sys.stdout.flush()
    test_acc = test(epoch, test_loader)
    best_acc = max(best_acc, test_acc)
    if epoch == cfg.epochs:
        print("Best Acc: %.2f Last Acc: %.2f" % (best_acc, test_acc))

    scheduler.step()

# record acc
if cfg.pretrained == True:
    torch.save(best_acc, './stage2/{}_pretrained_{}_{}_{}_best_acc.pt'.format(cfg.backbone, cfg.dataset, cfg.noise_mode, cfg.noise_ratio))
else:
    torch.save(best_acc, './stage2/{}_scratch_{}_{}_{}_best_acc.pt'.format(cfg.backbone, cfg.dataset, cfg.noise_mode, cfg.noise_ratio))