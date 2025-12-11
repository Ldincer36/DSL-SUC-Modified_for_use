import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

class SENET(nn.Module):
    def __init__(self, in_features, reduction=16):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(in_features, in_features // reduction, bias=False)
        self.fc2 = nn.Linear(in_features // reduction, in_features, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, l = x.shape
        y = self.global_avg_pool(x).view(b, c)
        y = F.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y)).view(b, c, 1)
        return x * y.expand_as(x)

