from data.load import load_imgs, get_loader, get_labels
from models.VGG import VGGSpecModel
from audtorch.metrics.functional import pearsonr
from torchvision.models import vgg16
import torch.nn as nn
import torch


# data preparation
print("reading data...")
BATCH = 32
data, ids = load_imgs()
valid_data = data[9000:]
valid_ids = ids[9000:]
valid_labels = get_labels(valid_ids)
valid_loader = get_loader(valid_data, valid_labels/9, batch_size=BATCH, shuffle=False)


# model preparation
print("setting up model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
val_model = VGGSpecModel(vgg16, 4096, 1, fcs=[256, 64]).half().to(device)
val_model.load_state_dict(torch.load(r"../trials/spec_valence1346.pth"))
val_model.eval()

aro_model = VGGSpecModel(vgg16, 4096, 1, fcs=[256, 64]).half().to(device)
aro_model.load_state_dict(torch.load(r"../trials/spec_arousal0941.pth"))
aro_model.eval()

dom_model = VGGSpecModel(vgg16, 4096, 1, fcs=[256, 64]).half().to(device)
dom_model.load_state_dict(torch.load(r"../trials/spec_dominance0957.pth"))
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
