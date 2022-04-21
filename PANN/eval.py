from data.load import get_labels, get_wav_fp, WAV_DIR
from data.dataset import LazyWavDataset
from torch.utils.data import Dataset, DataLoader
from audtorch.metrics.functional import pearsonr
from PANN.model import WaveNet
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch


# data preparation
print("reading data...")
BATCH = 16
data, ids = get_wav_fp()
test_data = data[9000:]
test_ids = ids[9000:]
test_labels = get_labels(test_ids)
test_ds = LazyWavDataset(test_data, test_labels/9)
test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False)

# model preparation
print("setting up model...")
# model preparation
sr = 22050
wsize, hsize, mel_bins = 520, 320, 128
fmin, fmax = 50, 8000
# fcs, dropout, act = [1024, 1024], 0.5, nn.ReLU
fcs, dropout, act = [2048], 0.5, nn.ReLU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
freeze = 10
val_model = WaveNet(1, 2048, sr=sr, wsize=wsize, hsize=hsize, mel_bins=mel_bins,
                    fmin=fmin, fmax=fmax, fcs=fcs, dropout=dropout, act=act, freeze=freeze).half().to(device)
val_model_fp = r"results/valence/models/freeze_none_valmse=0307.pth"
val_model.load_state_dict(torch.load(val_model_fp))
val_model.eval()

aro_model = WaveNet(1, 2048, sr=sr, wsize=wsize, hsize=hsize, mel_bins=mel_bins,
                    fmin=fmin, fmax=fmax, fcs=fcs, dropout=dropout, act=act, freeze=freeze).half().to(device)
aro_model_fp = r"results/arousal/models/freeze_none_valmse=0150.pth"
aro_model.load_state_dict(torch.load(aro_model_fp))
aro_model.eval()

dom_model = WaveNet(1, 2048, sr=sr, wsize=wsize, hsize=hsize, mel_bins=mel_bins,
                    fmin=fmin, fmax=fmax, fcs=fcs, dropout=dropout, act=act, freeze=freeze).half().to(device)
dom_model_fp = r"results/dominance/models/freeze_none_valmse=0165.pth"
dom_model.load_state_dict(torch.load(dom_model_fp))
dom_model.eval()


print("evaluating...")
mae, mse, pcc = nn.L1Loss(), nn.MSELoss(), lambda inputs, targets: pearsonr(inputs, targets)
preds, labels = None, None
with torch.no_grad():
    for x, y in test_loader:
        x = x.half().to(device)
        y = y.half().to(device)
        if labels is not None:
            labels = torch.cat((labels, y), dim=0)
        else:
            labels = y
        pred_val = val_model(x)
        pred_aro = aro_model(x)
        pred_dom = dom_model(x)
        batch_pred = torch.cat((pred_val, pred_aro, pred_dom), dim=1)
        if preds is not None:
            preds = torch.cat((preds, batch_pred), dim=0)
        else:
            preds = batch_pred
labels_val, preds_val = labels[:, 0], preds[:, 0]
labels_aro, preds_aro = labels[:, 1], preds[:, 1]
labels_dom, preds_dom = labels[:, 2], preds[:, 2]

# print(preds_val.size(), labels_val.size())
# exit(0)
val_mae, val_mse, val_pcc = mae(preds_val, labels_val), mse(preds_val, labels_val), pcc(preds_val, labels_val)
aro_mae, aro_mse, aro_pcc = mae(preds_aro, labels_aro), mse(preds_aro, labels_aro), pcc(preds_aro, labels_aro)
dom_mae, dom_mse, dom_pcc = mae(preds_dom, labels_dom), mse(preds_dom, labels_dom), pcc(preds_dom, labels_dom)
print(f"{val_mae.item():.5f}, {val_mse.item():.5f}, {val_pcc.item():.5f}")
print(f"{aro_mae.item():.5f}, {aro_mse.item():.5f}, {aro_pcc.item():.5f}")
print(f"{dom_mae.item():.5f}, {dom_mse.item():.5f}, {dom_pcc.item():.5f}")


fig, ax = plt.subplots()
ax.scatter(labels_val.cpu(), labels_val.cpu(), label="Ideal fit")
ax.scatter(labels_val.cpu(), preds_val.cpu(), label="Predictions", color='y')
ax.set_title("Valence Predictions against True Valence Level")
ax.set_xlabel("Valence Levels")
ax.set_ylabel("Predictions")
ax.legend()
plt.show()
