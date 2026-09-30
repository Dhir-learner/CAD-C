"""A small 3D ResNet for single-channel nodule patches."""
from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(cin, cout, 3, stride, 1, bias=False),
            nn.BatchNorm3d(cout),
            nn.ReLU(inplace=True),
            nn.Conv3d(cout, cout, 3, 1, 1, bias=False),
            nn.BatchNorm3d(cout),
        )
        self.shortcut = nn.Identity()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv3d(cin, cout, 1, stride, bias=False), nn.BatchNorm3d(cout))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.body(x) + self.shortcut(x))


class ResNet3D(nn.Module):
    """Outputs one logit per patch (malignant vs benign)."""

    def __init__(self, widths=(32, 64, 128, 256), blocks=(2, 2, 2, 2), dropout=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(1, widths[0], 3, 1, 1, bias=False),
            nn.BatchNorm3d(widths[0]),
            nn.ReLU(inplace=True),
        )
        layers, cin = [], widths[0]
        for stage, (width, n) in enumerate(zip(widths, blocks)):
            for b in range(n):
                layers.append(BasicBlock(cin, width, 2 if stage > 0 and b == 0 else 1))
                cin = width
        self.layers = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(cin, 1)
        )

    def forward(self, x):
        return self.head(self.layers(self.stem(x))).squeeze(1)
