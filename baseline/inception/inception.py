import torch.nn as nn


class InceptionClassifier(nn.Module):
    def __init__(self, inception_pretrained):
        super(InceptionClassifier, self).__init__()
        inception = inception_pretrained(pretrained=True, aux_logits=False)
        modules = list(inception.children())[:-1]
        self.resnet = nn.Sequential(*modules)

    def forward(self, x):
        return self.resnet(x)


class InceptionSpecModel(nn.Module):
    def __init__(self, inception_pretrained, out_dim,
                 fcs=[], dropout=0.2, act=nn.ReLU, init=nn.init.kaiming_normal_):
        super(InceptionSpecModel, self).__init__()
        inception = InceptionClassifier(inception_pretrained)
        self.inception = inception
        fcs = [2048] + fcs + [out_dim]
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
        inception_output = self.inception(x).reshape(-1, 2048)
        return self.classifier(inception_output)
