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
    def __init__(self, vgg_pretrained, embedding_dim, out_dim,
                 fcs=[], dropout=0.2, act=nn.ReLU, classifier_base=-1):
        super(VGGSpecModel, self).__init__()
        vgg = VGGClassifier(vgg_pretrained, classifier_base)
        self.vgg = vgg
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
        vgg_output = self.vgg(x)
        return self.classifier(vgg_output)
