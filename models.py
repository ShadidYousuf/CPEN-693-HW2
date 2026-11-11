from torch import nn


class FCNet(nn.Module):
    def __init__(self, hidden, activation, dropout=0.0):
        super().__init__()
        layers = [nn.Flatten()]
        size = 28 * 28
        for h in hidden:
            layers += [nn.Linear(size, h), activation(), nn.Dropout(dropout)]
            size = h
        layers.append(nn.Linear(size, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
