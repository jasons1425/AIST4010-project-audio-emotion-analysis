import torch.nn as nn
import numpy as np
import torch


# reference:
# https://github.com/harritaylor/torchvggish/blob/master/docs/_example_download_weights.ipynb
class VGGish(nn.Module):
    def __init__(self):
        super(VGGish, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),      # (64, 224, 224)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                # (64, 112, 112)
            nn.Conv2d(64, 128, 3, 1, 1),    # (128, 112, 112)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                # (128, 56, 56)
            nn.Conv2d(128, 256, 3, 1, 1),   # (256, 56, 56)
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),   # (256, 56, 56)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                # (256, 28, 28)
            nn.Conv2d(256, 512, 3, 1, 1),   # (512, 28, 28)
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),   # (512, 28, 28)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)                 # (512, 14, 14)
        )
        self.embedding = nn.Sequential(
            nn.Linear(512 * 196, 4096),     # (4096,)
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),          # (4096,)
            nn.ReLU(inplace=True),
            nn.Linear(4096, 128),           # (128,)
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        return x


class VGGishSpecModel(nn.Module):
    def __init__(self, embedding_dim, out_dim,
                 fcs=[], dropout=0.2, act=nn.ReLU):
        super(VGGishSpecModel, self).__init__()
        self.vggish = VGGish()
        fc_layers = []
        for idx in range(1, len(fcs)):
            fc_layers.append(nn.Linear(fcs[idx-1], fcs[idx]))
            fc_layers.append(act())
            fc_layers.append(nn.Dropout(dropout))
        if fcs:
            fc_layers.insert(0, nn.Linear(embedding_dim, fcs[0]))
            fc_layers.append(nn.Linear(fcs[-1], out_dim))
        else:
            fc_layers.append(nn.Linear(embedding_dim, out_dim))
        self.classifier = nn.Sequential(*fc_layers)

    def forward(self, x):
        vggish_output = self.vggish(x)
        return self.classifier(vggish_output)
