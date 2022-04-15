from data.load import load_imgs_png, get_loader, get_labels
from data.preprocess import spectrum_transform
from baseline.freqcnn.model import LargeConvSpecModel, LargeConvFEv1, LargeConvFEv3, LargeConvFEv7, \
    LargeConvFEv12, LargeConvFEv11
from baseline.densenet.densenet import DenseNetSpecModel
from audtorch.metrics.functional import pearsonr
from torchvision.models import densenet121
import torch.nn as nn
import matplotlib.pyplot as plt
import torch


# data preparation
print("reading data...")
BATCH = 32
data, ids = load_imgs_png()
valid_data = data[9000:]
valid_ids = ids[9000:]
valid_labels = get_labels(valid_ids)
# valid_transform = spectrum_transform(resize=224, norm=True, freq_mask=None, time_mask=None)
valid_transform = spectrum_transform(resize=128, norm=True, freq_mask=None, time_mask=None)
valid_loader = get_loader(valid_data, valid_labels/9, batch_size=BATCH, transform=valid_transform, shuffle=False)

# model preparation
print("setting up model...")
FC = [2048]
DROPOUT, CLS_BASE = 0.5, -1
# FE_MODEL, EMBEDDING_DIM = LargeConvFEv1, 101376  # embedding dim depends on fe_model
# FE_MODEL, EMBEDDING_DIM = LargeConvFEv3, 26880      # embedding dim depends on fe_model
# FE_MODEL, EMBEDDING_DIM = LargeConvFEv7, 23040     # embedding dim depends on fe_model
# FE_MODEL, EMBEDDING_DIM = LargeConvFEv11, 4096    # embedding dim depends on fe_model
FE_MODEL, EMBEDDING_DIM = LargeConvFEv12, 4096    # embedding dim depends on fe_model
device = "cuda" if torch.cuda.is_available() else "cpu"
val_model = LargeConvSpecModel(FE_MODEL, 3, 1, EMBEDDING_DIM,
                               fcs=FC, dropout=DROPOUT).half().to(device)
val_model.load_state_dict(torch.load(r"largeconv_v12_valence0319.pth"))
val_model.eval()

aro_model = LargeConvSpecModel(FE_MODEL, 3, 1, EMBEDDING_DIM,
                               fcs=FC, dropout=DROPOUT).half().to(device)
aro_model.load_state_dict(torch.load(r"largeconv_v12_arousal0152.pth"))
aro_model.eval()

dom_model = LargeConvSpecModel(FE_MODEL, 3, 1, EMBEDDING_DIM,
                               fcs=FC, dropout=DROPOUT).half().to(device)
dom_model.load_state_dict(torch.load(r"largeconv_v12_dominance0167.pth"))
dom_model.eval()


print("evaluating...")
mae, mse, pcc = nn.L1Loss(), nn.MSELoss(), lambda inputs, targets: pearsonr(inputs, targets)
preds, labels = None, None
with torch.no_grad():
    for x, y in valid_loader:
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
ax.set_title("LargeConv - Valence Predictions against True Valence Level")
ax.set_xlabel("Valence Levels")
ax.set_ylabel("Predictions")
ax.legend()
plt.show()
