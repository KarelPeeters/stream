import torch
from torch import nn


class ConvNetwork(nn.Module):
    def __init__(self, depth: int, n: int, c: int, s: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.convs = nn.Sequential(
            *(nn.Conv2d(c, c, 3, padding=1) for _ in range(depth))
        )

        self.input = torch.randn(n, c, s, s)

    def forward(self, x):
        return self.convs(x)

    def example_input(self):
        return self.input


class TestNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        n = 1
        c = 16
        s = 64

        self.linear_input = torch.randn(n, c)
        self.conv_input = torch.randn(n, c, s, s)

        self.left = nn.Sequential(
            nn.Linear(c, 2 * c),
            nn.Linear(2 * c, c),
            # nn.Linear(c, c),
            # nn.Linear(c, c),
        )
        self.right = nn.Sequential(
            nn.Linear(c, c),
            nn.Linear(c, c),
            # nn.Linear(c, c),
        )

        self.conv0 = nn.Conv2d(c, c, 3, padding=1)
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)

        self.conv = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.Conv2d(c, c, 3, padding=1),
            nn.Conv2d(c, c, 3, padding=1),
            nn.Conv2d(c, c, 3, padding=1),
            # nn.Conv2d(c, c, 3, padding=1),
            # nn.Conv2d(c, c, 3, padding=1),
            # nn.Conv2d(c, c, 3, padding=1),
            # nn.Conv2d(c, c, 3, padding=1),
        )

    def forward(self, x):
        return self.conv(x)

        # y = self.conv0(x)
        # z = self.conv1(y)
        #
        # return y, z
        # return z

        # return self.left(x)

    def example_input(self):
        return self.conv_input
        # return self.linear_input
