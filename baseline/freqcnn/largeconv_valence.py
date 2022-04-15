from data.load import load_imgs_png, get_loader, get_labels
from data.preprocess import spectrum_transform
from helper.process import train_model
from baseline.freqcnn.model import LargeConvFEv1, LargeConvSpecModel, \
    LargeConvFEv2, LargeConvFEv3, LargeConvFEv4, LargeConvFEv5, LargeConvFEv6, \
    LargeConvFEv7, LargeConvFEv8, LargeConvFEv9, LargeConvFEv10, LargeConvFEv11, \
    LargeConvFEv12
from torch.utils.data import random_split
import torch.nn as nn
import numpy as np
import torch


# data preparation
BATCH = 32
TRAIN_SIZE, VAL_SIZE = 8500, 500
data, ids = load_imgs_png()  # image channel is 3
train_data, valid_data = data[:TRAIN_SIZE], data[TRAIN_SIZE:TRAIN_SIZE+VAL_SIZE]
train_ids, valid_ids = ids[:TRAIN_SIZE], ids[TRAIN_SIZE:TRAIN_SIZE+VAL_SIZE]
train_labels = get_labels(train_ids)[:, 0].reshape(-1, 1) / 9  # map the 9-point scale to 0-1 scale
valid_labels = get_labels(valid_ids)[:, 0].reshape(-1, 1) / 9  # map the 9-point scale to 0-1 scale
# if resize is None, the image dimension will be 217 * 334 (H * W)
# train_transform = spectrum_transform(resize=224, norm=True, freq_mask=(0.5, 0.1), time_mask=(0.5, 0.1))
# valid_transform = spectrum_transform(resize=224, norm=True, freq_mask=None, time_mask=None)
train_transform = spectrum_transform(resize=128, norm=True)
valid_transform = spectrum_transform(resize=128, norm=True, freq_mask=None, time_mask=None)
train_loader = get_loader(train_data, train_labels, batch_size=BATCH,
                          transform=train_transform, shuffle=True)
valid_loader = get_loader(valid_data, valid_labels, batch_size=BATCH,
                          transform=valid_transform, shuffle=False)


# model preparation
FC = [2048]
DROPOUT, CLS_BASE = 0.5, -1
# FE_MODEL, EMBEDDING_DIM = LargeConvFEv1, 101376  # embedding dim depends on fe_model
# FE_MODEL, EMBEDDING_DIM = LargeConvFEv2, 10368   # embedding dim depends on fe_model
# FE_MODEL, EMBEDDING_DIM = LargeConvFEv3, 26880   # embedding dim depends on fe_model
# FE_MODEL, EMBEDDING_DIM = LargeConvFEv4, 24576   # embedding dim depends on fe_model
# FE_MODEL, EMBEDDING_DIM = LargeConvFEv5, 20736   # embedding dim depends on fe_model
# FE_MODEL, EMBEDDING_DIM = LargeConvFEv6, 26880   # embedding dim depends on fe_model
# FE_MODEL, EMBEDDING_DIM = LargeConvFEv8, 15360   # embedding dim depends on fe_model
# FE_MODEL, EMBEDDING_DIM = LargeConvFEv9, 111872  # embedding dim depends on fe_model
# FE_MODEL, EMBEDDING_DIM = LargeConvFEv10, 19712    # embedding dim depends on fe_model
# FE_MODEL, EMBEDDING_DIM = LargeConvFEv11, 4096    # embedding dim depends on fe_model, 8m52s
FE_MODEL, EMBEDDING_DIM = LargeConvFEv12, 4096    # embedding dim depends on fe_model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = LargeConvSpecModel(FE_MODEL, 3, 1, EMBEDDING_DIM,
                           fcs=FC, dropout=DROPOUT).half().to(device)

# training params
LR, MOMENTUM, DECAY = 1e-4, 0.9, 1e-3
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),
                            lr=LR, momentum=MOMENTUM, weight_decay=DECAY)


# training
EPOCHS = 60
best_model, losses = train_model(model, train_loader, criterion, optimizer, EPOCHS,
                                 device, valid_loader=valid_loader, half=True)
with open("largeconv_v12_nomask_valence_loss.npy", 'wb') as f:
    np.save(f, np.array(losses['valid']))
torch.save(model.state_dict(), f"largeconv_v12_nomask_valence.pth")
