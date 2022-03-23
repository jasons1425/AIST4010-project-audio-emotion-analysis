from data.load import load_imgs, get_loader, get_labels
from helper.process import train_model
from models.VGG import VGGSpecModel
from torchvision.models import vgg16, vgg11, vgg19, vgg13, vgg11_bn
import torch.nn as nn
import torch


# data preparation
BATCH = 64
data, ids = load_imgs()
train_data, valid_data = data[:8000], data[8000:9000]
train_ids, valid_ids = ids[:8000], ids[8000:9000]
train_labels, valid_labels = get_labels(train_ids), get_labels(valid_ids)
train_loader = get_loader(train_data, train_labels[:, 0].reshape(-1, 1)/9, batch_size=BATCH)
valid_loader = get_loader(valid_data, valid_labels[:, 0].reshape(-1, 1)/9, batch_size=BATCH)


# model preparation
FC = [256, 64]
DROPOUT, CLS_BASE = 0.5, -1
device = "cuda" if torch.cuda.is_available() else "cpu"
model = VGGSpecModel(vgg11, 4096, 1, fcs=FC, dropout=DROPOUT,
                     classifier_base=CLS_BASE).half().to(device)


# training params
LR, MOMENTUM, DECAY = 0.01, 0.9, 0.01
criterion = nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(),
                            lr=LR, momentum=MOMENTUM, weight_decay=DECAY)


# training
EPOCHS = 10
best_model, losses = train_model(model, train_loader, criterion, optimizer, EPOCHS,
                                 device, valid_loader=valid_loader, half=True)
torch.save(model.state_dict(), f"spec_valence.pth")
