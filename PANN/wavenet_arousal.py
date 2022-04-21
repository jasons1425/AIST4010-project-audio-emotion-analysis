from data.load import get_labels, get_wav_fp, WAV_DIR
from data.dataset import LazyWavDataset
from helper.process import train_model
from torch.utils.data import Dataset, DataLoader
from PANN.model import WaveNet
import torch.nn as nn
import numpy as np
import torch


# data preparation
BATCH = 16
TRAIN_SIZE, VAL_SIZE = 8500, 500
data, ids = get_wav_fp()  # image channel is 3
train_data, valid_data = data[:TRAIN_SIZE], data[TRAIN_SIZE:TRAIN_SIZE+VAL_SIZE]
train_ids, valid_ids = ids[:TRAIN_SIZE], ids[TRAIN_SIZE:TRAIN_SIZE+VAL_SIZE]
train_labels = get_labels(train_ids)[:, 1].reshape(-1, 1) / 9  # map the 9-point scale to 0-1 scale
valid_labels = get_labels(valid_ids)[:, 1].reshape(-1, 1) / 9  # map the 9-point scale to 0-1 scale
train_ds, valid_ds = LazyWavDataset(train_data, train_labels), LazyWavDataset(valid_data, valid_labels)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=BATCH, shuffle=False)


# model preparation
sr = 22050
wsize, hsize, mel_bins = 520, 320, 128
fmin, fmax = 50, 8000
# fcs, dropout, act = [1024, 1024], 0.5, nn.ReLU
fcs, dropout, act = [2048], 0.5, nn.ReLU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
freeze = 0
model = WaveNet(1, 2048, sr=sr, wsize=wsize, hsize=hsize, mel_bins=mel_bins,
                fmin=fmin, fmax=fmax, fcs=fcs, dropout=dropout, act=act, freeze=freeze).half().to(device)

# training params
# lr=1e-3, batch=16, 30 epochs, 160 m 34 s, 0.0150, fcs=[2048]
LR, MOMENTUM, DECAY = 1e-3, 0.9, 1e-3
criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=DECAY)
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=LR, momentum=MOMENTUM, weight_decay=DECAY)

# training
EPOCHS = 30
best_model, losses = train_model(model, train_loader, criterion, optimizer, EPOCHS,
                                 device, valid_loader=valid_loader, half=True)
with open("pann_arousal_loss.npy", 'wb') as f:
    np.save(f, np.array(losses['valid']))
torch.save(model.state_dict(), f"pann_arousal.pth")
