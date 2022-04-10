import torch
import torch.nn as nn
import torch.nn.functional as F


class EfficientNetClassifier(nn.Module):
    def __init__(self, efficientnet_pretrained):
        super(EfficientNetClassifier, self).__init__()
        efficientnet = efficientnet_pretrained(pretrained=True)
        efficientnet.classifier = nn.Sequential(
            # remove the dropout while being compatible with efficientnet forward() method
            nn.Dropout(p=0.0, inplace=True)
        )
        self.efficientnet = efficientnet

    def forward(self, x):
        out = self.efficientnet(x)
        return out


class EfficientNetSpecModel(nn.Module):
    def __init__(self, densenet_pretrained, embedding_dim, out_dim,
                 fcs=[], dropout=0.2, act=nn.ReLU, init=nn.init.kaiming_normal_):
        super(EfficientNetSpecModel, self).__init__()
        efficientnet = EfficientNetClassifier(densenet_pretrained)
        self.efficientnet = efficientnet
        fcs = [embedding_dim] + fcs + [out_dim]
        fc_layers = []
        for idx in range(1, len(fcs)):
            if idx != 1:
                fc_layers.append(nn.Dropout(dropout))
            fc_layers.append(nn.Linear(fcs[idx-1], fcs[idx]))
            if act and idx != (len(fcs) - 1):
                fc_layers.append(act())
        if init:
            for layer in fc_layers:
                if type(layer) == nn.Linear:
                    init(layer.weight)
        self.classifier = nn.Sequential(*fc_layers)

    def forward(self, x):
        efficientnet_output = self.efficientnet(x)
        return self.classifier(efficientnet_output)
