import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size=3, dropout_rate=0):
        super(DenseLayer, self).__init__()
        self.bn = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv1d(in_channels, growth_rate, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        out = self.dropout(out)
        out = torch.cat([x, out], dim=1)
        return out

class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, dropout_rate=0):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate,
                                     growth_rate,
                                     dropout_rate=dropout_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(self.relu(self.bn(x)))
        return self.pool(x)

class DenseNet1D(nn.Module):
    def __init__(
        self,
        in_channels=5,
        out_channels=64,
        num_blocks=1,
        num_layers_per_block=3,
        growth_rate=16,
        dropout_rate=0,
        add_transition_after_last_block=False
    ):
        super(DenseNet1D, self).__init__()

        self.initial_conv = nn.Conv1d(in_channels, growth_rate * 2, kernel_size=3,
                                      padding=1, bias=False)
        channels = growth_rate * 2
        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()

        for i in range(num_blocks):
            block = DenseBlock(
                num_layers=num_layers_per_block,
                in_channels=channels,
                growth_rate=growth_rate,
                dropout_rate=dropout_rate
            )
            self.blocks.append(block)
            channels += num_layers_per_block * growth_rate
            if i != num_blocks - 1 or add_transition_after_last_block:
                out_channels_tr = channels // 2
                self.transitions.append(TransitionLayer(channels, out_channels_tr))
                channels = out_channels_tr

        self.final_conv = nn.Conv1d (channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.initial_conv(x)
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i < len(self.transitions):
                x = self.transitions[i](x)
        x = self.final_conv(x)
        return x

