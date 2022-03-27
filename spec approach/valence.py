from data.load import load_imgs, get_loader, get_labels
from data.preprocess import spectrum_transform_crop
from helper.process import train_model
from models.VGG import VGGSpecModel, PlainCNN, AcousticSceneCNN
from models.VGGish import VGGishSpecModel
from torchvision.models import vgg16, vgg11, vgg19, vgg13, vgg11_bn
import torch.nn as nn
import torch


# data preparation
BATCH = 32
data, ids = load_imgs()
train_data, valid_data = data[:8000], data[8000:9000]
train_ids, valid_ids = ids[:8000], ids[8000:9000]
train_labels, valid_labels = get_labels(train_ids), get_labels(valid_ids)
train_loader = get_loader(train_data, train_labels[:, 0].reshape(-1, 1)/9,
                          batch_size=BATCH, transform=spectrum_transform_crop())
valid_loader = get_loader(valid_data, valid_labels[:, 0].reshape(-1, 1)/9,
                          batch_size=BATCH)


# model preparation
# FC = [256, 128]
# DROPOUT, CLS_BASE = 0.5, -1
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = VGGSpecModel(vgg16, 4096, 1, fcs=FC, dropout=DROPOUT,
#                      classifier_base=CLS_BASE).half().to(device)

# FC = [128, 128, 64]
# DROPOUT = 0.5
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = VGGishSpecModel(128, 1, fcs=FC, dropout=DROPOUT).half().to(device)

# FC = [1024, 256]
# DROPOUT = 0.5
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = PlainCNN(14400, 1, fcs=FC, dropout=DROPOUT).half().to(device)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AcousticSceneCNN().half().to(device)


# training params
LR, MOMENTUM, DECAY = 1e-3, 0.9, 0.01
criterion = nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(),
                            lr=LR, momentum=MOMENTUM, weight_decay=DECAY)


# training
EPOCHS = 50
best_model, losses = train_model(model, train_loader, criterion, optimizer, EPOCHS,
                                 device, valid_loader=valid_loader, half=True)
# torch.save(model.state_dict(), f"spec_valence.pth")
# torch.save(model.state_dict(), f"spec_vggish_valence.pth")
# torch.save(model.state_dict(), f"cnn_spec_valence.pth")
torch.save(model.state_dict(), f"acoucnn_spec_valence.pth")
