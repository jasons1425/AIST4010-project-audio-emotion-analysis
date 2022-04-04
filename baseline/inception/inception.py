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
        fc_layers = []
        for idx in range(1, len(fcs)):
            fc_layers.append(nn.Linear(fcs[idx-1], fcs[idx]))
            fc_layers.append(act())
            fc_layers.append(nn.Dropout(dropout))
        if fcs:
            fc_layers.insert(0, nn.Linear(2048, fcs[0]))
            fc_layers.append(nn.Linear(fcs[-1], out_dim))
        else:
            fc_layers.append(nn.Linear(2048, out_dim))
        if init:
            for layer in fc_layers:
                if type(layer) == nn.Linear:
                    init(layer.weight)
        self.classifier = nn.Sequential(*fc_layers)

    def forward(self, x):
        inception_output = self.inception(x).reshape(-1, 2048)
        return self.classifier(inception_output)
