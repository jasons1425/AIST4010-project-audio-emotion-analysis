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


class PlainCNN(nn.Module):
    def __init__(self, embedding_dim, out_dim,
                 fcs=[], dropout=0.2, act=nn.ReLU):
        super().__init__()
        # config A
        config = [
            nn.Conv2d(3, 32, 3, 1, 1),  # 32 * 224 * 224
            act(),
            nn.MaxPool2d(2),  # 32 * 112 * 112
            nn.Conv2d(32, 64, 3, 1, 1),  # 64 * 112 * 112
            act(),
            nn.MaxPool2d(2),  # 64 * 56 * 56
            nn.Conv2d(64, 128, 3, 1, 1),  # 128 * 56 * 56
            act(),
            nn.MaxPool2d(2),  # 128 * 28 * 28
            nn.Flatten()  # 100352
        ]
        # config B
        config = [
            nn.Conv2d(3, 32, 7),        # 32 * 218 * 218
            act(),
            nn.MaxPool2d(2),            # 32 * 109 * 109
            nn.Conv2d(32, 64, 5),       # 64 * 105 * 105
            act(),
            nn.MaxPool2d(2),            # 64 * 52 * 52
            nn.Conv2d(64, 128, 3),      # 128 * 50 * 50
            act(),
            nn.MaxPool2d(2),            # 128 * 25 * 25
            nn.Flatten()                # 80000
        ]
        # config C
        config = [
            nn.Conv2d(3, 32, 7),        # 32 * 218 * 218
            act(),
            nn.MaxPool2d(3),            # 32 * 72 * 72
            nn.Conv2d(32, 64, 5),       # 64 * 68 * 68
            act(),
            nn.MaxPool2d(2),            # 64 * 34 * 34
            nn.Conv2d(64, 64, 5),       # 64 * 30 * 30
            act(),
            nn.MaxPool2d(2),            # 64 * 15 * 15
            nn.Flatten()                # 14400
        ]
        self.conv_stack = nn.Sequential(*config)
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
        conv_output = self.conv_stack(x)
        fc_output = self.classifier(conv_output)
        return fc_output


# ref: https://arxiv.org/abs/1809.01543
class AcousticSceneCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # config D
        # ref: https://arxiv.org/abs/1809.01543
        config = [
            nn.Conv2d(3, 32, 5, 2, 2),      # 32 * 112 * 112
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),     # 32 * 112 * 112
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                # 32 * 56 * 56
            nn.Dropout(0.3),

            nn.Conv2d(32, 64, 3, 1, 1),     # 64 * 56 * 56
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),     # 64 * 56 * 56
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                # 64 * 28 * 28
            nn.Dropout(0.3),

            nn.Conv2d(64, 128, 3, 1, 1),    # 128 * 28 * 28
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),   # 128 * 28 * 28
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),   # 128 * 28 * 28
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),   # 128 * 28 * 28
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),                # 128 * 14 * 14
            nn.Dropout(0.3),

            nn.Conv2d(128, 512, 3, 1, 0),   # 512 * 12 * 12
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(512, 512, 1, 1, 0),   # 512 * 12 * 12
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(512, 1, 1, 1, 0),     # 1 * 12 * 12
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten()
        ]
        self.conv_stack = nn.Sequential(*config)
        self.fc = nn.Linear(12 * 12, 1)

    def forward(self, x):
        conv_output = self.conv_stack(x)
        fc_output = self.fc(conv_output)
        return fc_output
