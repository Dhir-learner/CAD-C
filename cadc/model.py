"""Nodule classifiers: a small 3D ResNet, and a 2.5D multi-view CNN with an ImageNet-pretrained backbone.

Both take a (B, 1, D, H, W) cube scaled to [0, 1] and return one logit per cube.
"""
import torch
import torch.nn.functional as F
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


class MultiView2D(nn.Module):
    """Looks at 9 planes through the cube's centre (3 orthogonal + 6 diagonal) with one shared
    ImageNet-pretrained ResNet18, then pools the per-view features (mean and max) into one logit.
    """

    def __init__(self, pretrained=True, size=96, dropout=0.3):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        net.fc = nn.Identity()
        self.backbone = net
        self.size = size
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(512 * 2, 1))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @staticmethod
    def views(x):
        """(B, 1, D, H, W) -> (B, 9, S, S) planes through the centre; assumes a cube."""
        c = x.shape[2] // 2
        planes = [x[:, 0, c], x[:, 0, :, c], x[:, 0, :, :, c]]
        # Diagonal planes: keep one axis, walk the diagonal and anti-diagonal of the other two.
        for keep, (d1, d2) in enumerate([(3, 4), (2, 4), (2, 3)]):
            for flip in (False, True):
                v = torch.flip(x, dims=(d2,)) if flip else x
                diag = torch.diagonal(v[:, 0], dim1=d1 - 1, dim2=d2 - 1)  # diagonal moves to the last dim
                planes.append(diag)
        return torch.stack(planes, dim=1)

    def forward(self, x):
        b = x.shape[0]
        v = self.views(x)
        n = v.shape[1]
        v = F.interpolate(v.reshape(b * n, 1, *v.shape[2:]), size=(self.size, self.size), mode="bilinear", align_corners=False)
        v = (v.expand(-1, 3, -1, -1) - self.mean) / self.std
        f = self.backbone(v).view(b, n, -1)
        return self.head(torch.cat([f.mean(1), f.amax(1)], dim=1)).squeeze(1)


ARCHS = {"resnet3d": ResNet3D, "multiview2d": MultiView2D}
DEFAULT_LR = {"resnet3d": 1e-3, "multiview2d": 3e-4}


def build_model(arch, pretrained=True):
    if arch == "multiview2d":
        return MultiView2D(pretrained=pretrained)
    return ARCHS[arch]()
