import torch.nn as nn


class VGGClassifier(nn.Module):
    def __init__(self, vgg_pretrained, classifier_base):
        super(VGGClassifier, self).__init__()
        vgg = vgg_pretrained(pretrained=True)
        vgg.classifier = vgg.classifier[:classifier_base]
        self.vgg = vgg

    def forward(self, x):
        return self.vgg(x)


class VGGSpecModel(nn.Module):
    def __init__(self, vgg_pretrained, fe_dim, out_dim,
                 fcs=[], dropout=0.2, act=nn.ReLU, classifier_base=-1,
                 init=nn.init.kaiming_normal_):
        super(VGGSpecModel, self).__init__()
        vgg = VGGClassifier(vgg_pretrained, classifier_base)
        self.vgg = vgg
        fcs = [fe_dim] + fcs + [out_dim]
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
        vgg_output = self.vgg(x)
        return self.classifier(vgg_output)
