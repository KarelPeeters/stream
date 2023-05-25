import torch
from torch import nn


class TestNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        n = 512
        c = 64
        s = 32

        self.linear_input = torch.randn(n, c)
        self.conv_input = torch.randn(1, c, s, s)

        self.left = nn.Sequential(
            nn.Linear(c, c),
            nn.Linear(c, c),
            # nn.Linear(c, c),
            # nn.Linear(c, c),
        )
        self.right = nn.Sequential(
            nn.Linear(c, c),
            nn.Linear(c, c),
            # nn.Linear(c, c),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.Conv2d(64, 64, 3, padding=1),
        )

    def forward(self, x):
        # return self.conv(x)
        return self.left(x)

    def example_input(self):
        # return self.conv_input
        return self.linear_input
