import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseNetClassifier(nn.Module):
    def __init__(self, densenet_pretrained):
        super(DenseNetClassifier, self).__init__()
        densenet = densenet_pretrained(pretrained=True)
        modules = list(densenet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

    def forward(self, x):
        out = self.resnet(x)
        out = F.relu(out, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return out


class DenseNetSpecModel(nn.Module):
    def __init__(self, densenet_pretrained, embedding_dim, out_dim,
                 fcs=[], dropout=0.2, act=nn.ReLU, init=nn.init.kaiming_normal_):
        super(DenseNetSpecModel, self).__init__()
        densenet = DenseNetClassifier(densenet_pretrained)
        self.densenet = densenet
        self.embedding_dim = embedding_dim
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
        if init:
            for layer in fc_layers:
                if type(layer) == nn.Linear:
                    init(layer.weight)
        self.classifier = nn.Sequential(*fc_layers)

    def forward(self, x):
        densenet_output = self.densenet(x).reshape(-1, self.embedding_dim)
        return self.classifier(densenet_output)
