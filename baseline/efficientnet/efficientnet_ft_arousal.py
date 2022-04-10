from data.load import load_imgs_png, get_loader, get_labels
from data.preprocess import spectrum_transform
from helper.process import train_model
from baseline.efficientnet.efficientnet import EfficientNetSpecModel
from torch.utils.data import random_split
from torchvision.models import efficientnet_b2
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
train_labels = get_labels(train_ids)[:, 1].reshape(-1, 1) / 9  # map the 9-point scale to 0-1 scale
valid_labels = get_labels(valid_ids)[:, 1].reshape(-1, 1) / 9  # map the 9-point scale to 0-1 scale
# if resize is None, the image dimension will be 217 * 334 (H * W)
train_transform = spectrum_transform(resize=288, norm=True)
valid_transform = spectrum_transform(resize=288, norm=True, freq_mask=None, time_mask=None)
train_loader = get_loader(train_data, train_labels, batch_size=BATCH,
                          transform=train_transform, shuffle=True)
valid_loader = get_loader(valid_data, valid_labels, batch_size=BATCH,
                          transform=valid_transform, shuffle=False)


# model preparation
FC = [2048]
DROPOUT, CLS_BASE = 0.5, -1
device = "cuda" if torch.cuda.is_available() else "cpu"
# use SiLU as activation since it is used in the original paper
model = EfficientNetSpecModel(efficientnet_b2, 1408, 1, fcs=FC, dropout=DROPOUT).half().to(device)

# training params
LR, MOMENTUM, DECAY = 1e-4, 0.9, 1e-3
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),
                            lr=LR, momentum=MOMENTUM, weight_decay=DECAY)


# training
EPOCHS = 20
best_model, losses = train_model(model, train_loader, criterion, optimizer, EPOCHS,
                                 device, valid_loader=valid_loader, half=True)
with open("efficientnet_arousal_loss.npy", 'wb') as f:
    np.save(f, np.array(losses['valid']))
torch.save(model.state_dict(), f"efficientnet_arousal.pth")
