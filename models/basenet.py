import torch.nn as nn


class Basenet(nn.Module):
    def __init__(self):
        super(Basenet, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, padding=1)
        )

    def forward(self, x):
        return self.main(x)
