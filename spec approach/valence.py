from data.load import load_imgs, load_imgs_png, get_loader, get_labels
from data.preprocess import spectrum_transform
from helper.process import train_model
from models.VGG import VGGSpecModel, PlainCNN, AcousticSceneCNN
from models.VGGish import VGGishSpecModel
from torchvision.models import vgg16, vgg11, vgg19, vgg13, vgg11_bn, alexnet
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
import time


# data preparation
BATCH = 32
since = time.time()
# data, ids = load_imgs()
data, ids = load_imgs_png()  # image channel is 1
train_data, valid_data = data[:8000], data[8000:9000]
train_ids, valid_ids = ids[:8000], ids[8000:9000]
train_labels, valid_labels = get_labels(train_ids)[:, 0].reshape(-1, 1), get_labels(valid_ids)[:, 0].reshape(-1, 1)
mm_scaler = MinMaxScaler(feature_range=(0, 1))
train_labels = mm_scaler.fit_transform(train_labels)
valid_labels = mm_scaler.transform(valid_labels)
# if resize is None, the image dimension will be 217 * 334 (H * W)
train_loader = get_loader(train_data, train_labels, batch_size=BATCH,
                          transform=spectrum_transform(resize=None, norm=None), shuffle=True)
valid_loader = get_loader(valid_data, valid_labels, batch_size=BATCH,
                          transform=spectrum_transform(resize=None, norm=None), shuffle=False)


# model preparation
# FC = [128, 64]
# DROPOUT, CLS_BASE = 0.5, -1
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = VGGSpecModel(vgg11, 4096, 1, fcs=FC, dropout=DROPOUT,
#                      classifier_base=CLS_BASE).half().to(device)

# FC = [128, 128, 64]
# DROPOUT = 0.5
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = VGGishSpecModel(128, 1, fcs=FC, dropout=DROPOUT).half().to(device)

FC = [128, 128]
DROPOUT = 0.5
CLASSIFIER_FUNC = None
IN_DIM = 3
device = "cuda" if torch.cuda.is_available() else "cpu"
model = PlainCNN(28672, 1, in_dim=IN_DIM, fcs=FC, dropout=DROPOUT,
                 classifier_func=CLASSIFIER_FUNC).half().to(device)

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = AcousticSceneCNN().half().to(device)


# training params
LR, MOMENTUM, DECAY = 1e-3, 0.9, 0.01
criterion = nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(),
                            lr=LR, momentum=MOMENTUM, weight_decay=DECAY)


# training
EPOCHS = 10
best_model, losses = train_model(model, train_loader, criterion, optimizer, EPOCHS,
                                 device, valid_loader=valid_loader, half=True)
# torch.save(model.state_dict(), f"spec_valence.pth")
# torch.save(model.state_dict(), f"spec_vggish_valence.pth")
torch.save(model.state_dict(), f"cnn_spec_valence.pth")
# torch.save(model.state_dict(), f"acoucnn_spec_valence.pth")
