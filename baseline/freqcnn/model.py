import torch.nn as nn
import torch
import math


class BnConv(nn.Module):
    def __init__(self, in_dim, out_dim, k_size, stride=1, padding=0):
        super().__init__()
        self.bn_conv = nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.Conv2d(in_dim, out_dim, k_size, stride, padding),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.bn_conv(x)


def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


class LargeConvFEv1(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        # expect input dimension to be (224 * 224)
        self.layers = nn.Sequential(
            BnConv(in_dim, 32, (55, 25), 1, 0),              # 32  * 170 * 200
            nn.AvgPool2d(2),                                 # 32  * 85  * 100
            BnConv(32, 128, 5, 1, 0),                        # 128 * 81  * 96
            nn.AvgPool2d(2),                                 # 128 * 40  * 48
            BnConv(128, 256, 3, 1, 0),                       # 256 * 38  * 46
            BnConv(256, 256, 3, 1, 0),                       # 256 * 36  * 44
            nn.AvgPool2d(2),                                 # 256 * 18  * 22
            nn.Flatten()
        )

    def forward(self, x):
        fe = self.layers(x)
        return fe


class LargeConvFEv2(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        # expect input dimension to be (224 * 224)
        self.layers = nn.Sequential(
            BnConv(in_dim, 16, (113, 23), (1, 2), 0),       # 16  * 112 * 101
            nn.AvgPool2d(2),                                # 16  * 56  * 50
            BnConv(16, 32, (11, 5), 1, 0),                  # 32  * 46  * 46
            nn.AvgPool2d(2),                                # 32  * 23  * 23
            BnConv(32, 128, 3, 1, 0),                       # 32  * 21  * 21
            BnConv(128, 128, 3, 1, 0),                      # 128 * 19  * 19
            nn.AvgPool2d(2),                                # 128 * 9  * 9
            nn.Flatten()
        )

    def forward(self, x):
        fe = self.layers(x)
        return fe


class LargeConvFEv3(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        # expect input dimension to be (128 * 128)
        self.layers = nn.Sequential(
            BnConv(in_dim, 32, (128, 3), 1, (32, 1)),   # 32  * 65  * 128
            nn.MaxPool2d(2),                            # 32  * 32  * 64
            BnConv(32, 64, 5, 1, 1),                    # 64  * 30  * 62
            BnConv(64, 64, 5, 1, 1),                    # 64  * 28  * 60
            nn.MaxPool2d(2),                            # 64  * 14  * 30
            BnConv(64, 256, 3, 1, 1),                   # 256 * 14  * 30
            BnConv(256, 256, 3, 1, 1),                  # 256 * 14  * 30
            nn.MaxPool2d(2),                            # 256 * 7  * 15
            nn.Flatten()                                # 26880
        )

    def forward(self, x):
        fe = self.layers(x)
        return fe


class LargeConvFEv4(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        # expect input dimension to be (128 * 128)
        self.layers = nn.Sequential(
            BnConv(in_dim, 32, (9, 41), 1, 0),          # 32  * 120 * 88
            nn.MaxPool2d(2),                            # 32  * 60  * 44
            BnConv(32, 64, 5, 1, 1),                    # 64  * 58  * 42
            BnConv(64, 64, 5, 1, 1),                    # 64  * 56  * 40
            nn.MaxPool2d(2),                            # 64  * 28  * 20
            BnConv(64, 256, 3, 1, 0),                   # 256 * 26  * 18
            BnConv(256, 256, 3, 1, 0),                  # 256 * 24  * 16
            nn.MaxPool2d(2),                            # 256 * 12  * 8
            nn.Flatten()                                # 24576
        )

    def forward(self, x):
        fe = self.layers(x)
        return fe


class LargeConvFEv5(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        # expect input dimension to be (128 * 128)
        self.layers = nn.Sequential(
            BnConv(in_dim, 32, 21, 1, 0),               # 32  * 108 * 108
            nn.MaxPool2d(2),                            # 32  * 54  * 54
            BnConv(32, 64, 5, 1, 0),                    # 64  * 50  * 50
            BnConv(64, 64, 5, 1, 0),                    # 64  * 46  * 46
            nn.MaxPool2d(2),                            # 64  * 23  * 23
            BnConv(64, 256, 3, 1, 0),                   # 256 * 21  * 21
            BnConv(256, 256, 3, 1, 0),                  # 256 * 19  * 19
            nn.MaxPool2d(2),                            # 256 * 9  * 9
            nn.Flatten()                                # 20736
        )

    def forward(self, x):
        fe = self.layers(x)
        return fe


class LargeConvFEv6(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        # expect input dimension to be (128 * 128)
        self.layers = nn.Sequential(
            BnConv(in_dim, 32, (3, 128), 1, (1, 32)),   # 32  * 128 * 65
            nn.MaxPool2d(2),                            # 32  * 64  * 32
            BnConv(32, 64, 5, 1, 1),                    # 64  * 62  * 30
            BnConv(64, 64, 5, 1, 1),                    # 64  * 60  * 28
            nn.MaxPool2d(2),                            # 64  * 30  * 14
            BnConv(64, 256, 3, 1, 1),                   # 256 * 30  * 14
            BnConv(256, 256, 3, 1, 1),                  # 256 * 30  * 14
            nn.MaxPool2d(2),                            # 256 * 15  * 14
            nn.Flatten()                                # 26880
        )

    def forward(self, x):
        fe = self.layers(x)
        return fe


class LargeConvFEv7(nn.Module):
    def __init__(self, in_dim, device=None, half=True):
        super().__init__()
        # expect input dimension to be (128 * 128)
        self.layers = nn.Sequential(
            BnConv(in_dim + 1, 32, (33, 21), 1, 0),         # 32  * 96  * 108
            nn.AvgPool2d(2),                                # 32  * 48  * 54
            BnConv(32, 128, 5, 1, 0),                       # 128 * 44  * 50
            nn.AvgPool2d(2),                                # 128 * 22  * 25
            BnConv(128, 256, 3, 1, 0),                      # 256 * 20  * 23
            BnConv(256, 256, 3, 1, 0),                      # 256 * 18  * 21
            nn.AvgPool2d(2),                                # 256 * 9   * 10
            nn.Flatten(),                                   # 23040
        )
        self.pos_enc = positionalencoding1d(128, 128)
        if half:
            self.pos_enc = self.pos_enc.half()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.pos_enc = self.pos_enc.to(device)

    def forward(self, x):
        batch_size = x.size(0)
        pos_enc = self.pos_enc.repeat([batch_size, 1, 1, 1])
        encoded_x = torch.cat((pos_enc, x), dim=1)     # channel depth becomes 4
        fe = self.layers(encoded_x)
        return fe


class LargeConvFEv8(nn.Module):
    def __init__(self, in_dim, device=None, half=True):
        super().__init__()
        # expect input dimension to be (128 * 128)
        self.layers = nn.Sequential(
            BnConv(in_dim + 1, 32, (9, 128), 1, (0, 32)),   # 32  * 120 * 65
            nn.AvgPool2d(2),                                # 32  * 60  * 32
            BnConv(32, 128, 5, 1, 0),                       # 128 * 56  * 28
            nn.AvgPool2d(2),                                # 128 * 28  * 14
            BnConv(128, 256, 3, 1, 0),                      # 256 * 26  * 12
            BnConv(256, 256, 3, 1, 0),                      # 256 * 24  * 10
            nn.AvgPool2d(2),                                # 256 * 12  * 5
            nn.Flatten(),                                   # 15360
        )
        self.pos_enc = positionalencoding1d(128, 128)       # add 'where the frequency is' info to x-axis
        if half:
            self.pos_enc = self.pos_enc.half()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.pos_enc = self.pos_enc.to(device)

    def forward(self, x):
        batch_size = x.size(0)
        pos_enc = self.pos_enc.repeat([batch_size, 1, 1, 1])
        encoded_x = torch.cat((pos_enc, x), dim=1)     # channel depth becomes 4
        fe = self.layers(encoded_x)
        return fe


class LargeConvFEv9(nn.Module):
    def __init__(self, in_dim, device=None, half=True):
        super().__init__()
        # expect input dimension to be (224 * 224)
        self.layers = nn.Sequential(
            BnConv(in_dim + 1, 32, (13, 45), 1, 0),         # 32  * 212 * 180
            nn.AvgPool2d(2),                                # 32  * 106 * 90
            BnConv(32, 128, 5, 1, 0),                       # 128 * 102 * 86
            nn.AvgPool2d(2),                                # 128 * 51  * 43
            BnConv(128, 256, 3, 1, 0),                      # 256 * 49  * 41
            BnConv(256, 256, 3, 1, 0),                      # 256 * 47  * 39
            nn.AvgPool2d(2),                                # 256 * 23  * 19
            nn.Flatten(),                                   # 111872
        )
        self.pos_enc = positionalencoding1d(224, 224)
        if half:
            self.pos_enc = self.pos_enc.half()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.pos_enc = self.pos_enc.to(device)

    def forward(self, x):
        batch_size = x.size(0)
        pos_enc = self.pos_enc.repeat([batch_size, 1, 1, 1])
        encoded_x = torch.cat((pos_enc, x), dim=1)     # channel depth becomes 4
        fe = self.layers(encoded_x)
        return fe


class LargeConvFEv10(nn.Module):
    def __init__(self, in_dim, device=None, half=True):
        super().__init__()
        # expect input dimension to be (128 * 128)
        self.layers = nn.Sequential(
            BnConv(in_dim + 1, 32, (13, 45), 1, 0),         # 32  * 116 * 84
            nn.AvgPool2d(2),                                # 32  * 58  * 42
            BnConv(32, 128, 5, 1, 0),                       # 128 * 54  * 38
            nn.AvgPool2d(2),                                # 128 * 27  * 19
            BnConv(128, 256, 3, 1, 0),                      # 256 * 25  * 17
            BnConv(256, 256, 3, 1, 0),                      # 256 * 23  * 15
            nn.AvgPool2d(2),                                # 256 * 11  * 7
            nn.Flatten(),                                   # 19712
        )
        self.pos_enc = positionalencoding1d(128, 128)
        if half:
            self.pos_enc = self.pos_enc.half()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.pos_enc = self.pos_enc.to(device)

    def forward(self, x):
        batch_size = x.size(0)
        pos_enc = self.pos_enc.repeat([batch_size, 1, 1, 1])
        encoded_x = torch.cat((pos_enc, x), dim=1)     # channel depth becomes 4
        fe = self.layers(encoded_x)
        return fe


class LargeConvFEv11(nn.Module):
    def __init__(self, in_dim, device=None, half=True):
        super().__init__()
        # expect input dimension to be (128 * 128)
        self.layers = nn.Sequential(
            BnConv(in_dim + 1, 32, (13, 29), 1, 0),         # 32  * 116 * 100
            nn.AvgPool2d(2),                                # 32  * 52  * 44
            BnConv(32, 64, 5, 1, 2),                        # 128 * 52  * 44
            BnConv(64, 64, 5, 1, 2),                        # 128 * 52  * 44
            nn.AvgPool2d(2),                                # 128 * 26  * 22
            BnConv(64, 256, 3, 1, 1),                       # 256 * 26  * 22
            BnConv(256, 256, 3, 1, 1),                      # 256 * 26  * 22
            nn.AvgPool2d(2),                                # 256 * 13  * 11
            BnConv(256, 1024, 3, 1, 0),                     # 1024 * 11  * 9
            BnConv(1024, 1024, 3, 1, 0),                    # 1024 * 9   * 7
            nn.AdaptiveAvgPool2d((2, 2)),                   # 1024 * 2   * 2
            nn.Flatten(),                                   # 4096
        )
        self.pos_enc = positionalencoding1d(128, 128)
        if half:
            self.pos_enc = self.pos_enc.half()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.pos_enc = self.pos_enc.to(device)

    def forward(self, x):
        batch_size = x.size(0)
        pos_enc = self.pos_enc.repeat([batch_size, 1, 1, 1])
        encoded_x = torch.cat((pos_enc, x), dim=1)     # channel depth becomes 4
        fe = self.layers(encoded_x)
        return fe


class LargeConvFEv12(nn.Module):
    def __init__(self, in_dim, device=None, half=True):
        super().__init__()
        # expect input dimension to be (128 * 128)
        self.layers = nn.Sequential(
            BnConv(in_dim, 32, (13, 29), 1, 0),         # 32  * 116 * 100
            nn.AvgPool2d(2),                                # 32  * 52  * 44
            BnConv(32, 64, 5, 1, 2),                        # 128 * 52  * 44
            BnConv(64, 64, 5, 1, 2),                        # 128 * 52  * 44
            nn.AvgPool2d(2),                                # 128 * 26  * 22
            BnConv(64, 256, 3, 1, 1),                       # 256 * 26  * 22
            BnConv(256, 256, 3, 1, 1),                      # 256 * 26  * 22
            nn.AvgPool2d(2),                                # 256 * 13  * 11
            BnConv(256, 1024, 3, 1, 0),                     # 1024 * 11  * 9
            BnConv(1024, 1024, 3, 1, 0),                    # 1024 * 9   * 7
            nn.AdaptiveAvgPool2d((2, 2)),                   # 1024 * 2   * 2
            nn.Flatten(),                                   # 4096
        )

    def forward(self, x):
        return self.layers(x)


class LargeConvSpecModel(nn.Module):
    def __init__(self, fe_model, in_dim, out_dim, fe_dim,
                 fcs=[], dropout=0.2, act=nn.ReLU, init=nn.init.kaiming_normal_):
        super().__init__()
        self.fe = fe_model(in_dim)
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
        spectrum_features = self.fe(x)
        return self.classifier(spectrum_features)


