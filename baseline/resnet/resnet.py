import torch.nn as nn


class ResNetClassifier(nn.Module):
    def __init__(self, resnet_pretrained):
        super(ResNetClassifier, self).__init__()
        resnet = resnet_pretrained(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

    def forward(self, x):
        return self.resnet(x)


class ResNetSpecModel(nn.Module):
    def __init__(self, resnet_pretrained, embedding_dim, out_dim,
                 fcs=[], dropout=0.2, act=nn.ReLU, init=nn.init.kaiming_normal_):
        super(ResNetSpecModel, self).__init__()
        resnet = ResNetClassifier(resnet_pretrained)
        self.resnet = resnet
        self.embedding_dim = embedding_dim
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
        resnet_output = self.resnet(x).reshape(-1, self.embedding_dim)
        return self.classifier(resnet_output)
