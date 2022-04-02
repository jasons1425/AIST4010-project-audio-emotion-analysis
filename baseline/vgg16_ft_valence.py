from data.load import load_imgs_png, get_loader, get_labels
from data.preprocess import spectrum_transform
from helper.process import train_model
from models.VGG import VGGSpecModel
from torch.utils.data import random_split
from torchvision.models import vgg16, vgg11, vgg19, vgg13, vgg11_bn, alexnet
import torch.nn as nn
import numpy as np
import torch


# data preparation
BATCH = 32
TRAIN_SIZE, VAL_SIZE = 8500, 500
data, ids = load_imgs_png()  # image channel is 3
train_idxs, val_idxs = random_split(np.arange(TRAIN_SIZE + VAL_SIZE), [TRAIN_SIZE, VAL_SIZE])
train_data, valid_data = data[train_idxs], data[val_idxs]
train_ids, valid_ids = ids[train_idxs], ids[val_idxs]
train_labels = get_labels(train_ids)[:, 0].reshape(-1, 1) / 9  # map the 9-point scale to 0-1 scale
valid_labels = get_labels(valid_ids)[:, 0].reshape(-1, 1) / 9  # map the 9-point scale to 0-1 scale
# if resize is None, the image dimension will be 217 * 334 (H * W)
train_loader = get_loader(train_data, train_labels, batch_size=BATCH,
                          transform=spectrum_transform(resize=224, norm=True), shuffle=True)
valid_loader = get_loader(valid_data, valid_labels, batch_size=BATCH,
                          transform=spectrum_transform(resize=224, norm=True), shuffle=False)


# model preparation
# 1024, 1024 - result: 0.0359
# 2048, 2048 - result: 0.0346
# 4096, 4096 - result: 0.0376
# 2048, 256  - result: 0.0356
# 256, 64    - result: 0.0501
FC = [256, 64]
DROPOUT, CLS_BASE = 0.5, -1
device = "cuda" if torch.cuda.is_available() else "cpu"
model = VGGSpecModel(vgg16, 4096, 1, fcs=FC, dropout=DROPOUT,
                     classifier_base=CLS_BASE).half().to(device)


# training params
LR, MOMENTUM, DECAY = 1e-4, 0.9, 1e-3
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),
                            lr=LR, momentum=MOMENTUM, weight_decay=DECAY)


# training
EPOCHS = 10
best_model, losses = train_model(model, train_loader, criterion, optimizer, EPOCHS,
                                 device, valid_loader=valid_loader, half=True)
torch.save(model.state_dict(), f"vgg16baseline_valence.pth")