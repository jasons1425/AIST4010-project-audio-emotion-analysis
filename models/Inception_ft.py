import torch.nn as nn
import torchvision.models as models


class InceptionFT(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # expect input shape to be (3, 299, 299)
        self.model = models.inception_v3(pretrained=pretrained, aux_logits=False)
        self.model.fc = nn.Linear(2048, 1)

    def forward(self, x):
        output = self.model(x)
        return output
