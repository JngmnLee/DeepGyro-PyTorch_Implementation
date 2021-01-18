import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class DeepBlind(nn.Module):
    def __init__(self, img_rows=256, img_cols=256):
        super(DeepBlind, self).__init__()

        self.img_rows = img_rows
        self.img_cols = img_cols

        # Encoder
        self.layer1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.layer3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.layer4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 512, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 512, kernel_size=3),
            nn.ReLU(inplace=True)
        )

        self.layer5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 1024, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(1024, 1024, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        )

        # Decoder
        self.layer6 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1024, 512, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 512, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        )

        self.layer7 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        )

        self.layer8 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        )

        self.layer9 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(inplace=True)
        )

        self.layer10 = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):

        # Encoder
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)

        # Decoder
        x6 = self.layer6(torch.cat((x5, x4), dim=1))
        x7 = self.layer7(torch.cat((x6, x3), dim=1))
        x8 = self.layer8(torch.cat((x7, x2), dim=1))
        x9 = self.layer9(torch.cat((x8, x1), dim=1))

        deblur_image = self.layer10(x9)

        return deblur_image


class DeepGyro(nn.Module):
    def __init__(self, img_rows = 256, img_cols=256):
        super(DeepGyro, self).__init__()

        self.img_rows = img_rows
        self.img_cols = img_cols

        # Encoder
        self.layer1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.layer3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.layer4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 512, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 512, kernel_size=3),
            nn.ReLU(inplace=True)
        )

        self.layer5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 1024, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(1024, 1024, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        )

        # Decoder
        self.layer6 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1024, 512, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 512, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        )

        self.layer7 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        )

        self.layer8 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        )

        self.layer9 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(inplace=True)
        )

        self.layer10 = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):

        # Encoder
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)

        # Decoder
        x6 = self.layer6(torch.cat((x5, x4), dim=1))
        x7 = self.layer7(torch.cat((x6, x3), dim=1))
        x8 = self.layer8(torch.cat((x7, x2), dim=1))
        x9 = self.layer9(torch.cat((x8, x1), dim=1))

        deblur_image = self.layer10(x9)

        return deblur_image
