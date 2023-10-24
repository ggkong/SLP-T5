import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(1, 8)),
            nn.ReLU(inplace=True),
        )
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=2)
        self.classifier = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 4),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(-1, 128, 8)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.squeeze(3)
        x = self.encoder_layer(x)
        x = torch.mean(x, axis=1)
        x = self.classifier(x)
        return x