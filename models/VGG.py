import torch.nn as nn


class VGGClassifier(nn.Module):
    def __init__(self, vgg_pretrained):
        super(VGGClassifier, self).__init__()
        vgg = vgg_pretrained(pretrained=True)
        vgg.classifier = vgg.classifier[:-1]
        self.vgg = vgg

    def forward(self, x):
        return self.vgg(x)


class VGGSpecModel(nn.Module):
    def __init__(self, vgg_pretrained, embedding_dim):
        super(VGGSpecModel, self).__init__()
        vgg = VGGClassifier(vgg_pretrained)
        self.vgg = vgg
        self.classifier = nn.Linear(embedding_dim, 3)

    def forward(self, x):
        vgg_output = self.vgg(x)
        return self.classifier(vgg_output)
