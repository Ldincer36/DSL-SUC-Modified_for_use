from typing import Dict, Any
from models.MLPs import MLP
from models.Densenet import *
import yaml
from models.senet import *
# 读取配置文件
with open("E:\study\DLS-SUC\config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

class  classifier(nn.Module):
    def __init__(
        self,
        in_channels,
        num_linears=[64],
        dropout_linears=[0.2],
        seqlen=33,
    ) -> None:
        super().__init__()
        self.DenseNet1D = DenseNet1D(
            in_channels=in_channels,
            out_channels=163,
            num_blocks=1,
            num_layers_per_block=3,
            growth_rate=8,
            dropout_rate=0.4,add_transition_after_last_block=False
        )

        self.lstm = nn.LSTM(
            input_size=163,
            hidden_size=96,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        self.senet = SENET(96*2)
        input_linear = 96*2 *33 # seqlen=54 时为 10368
        self.mlp = MLP(input_linear, num_linears, dropout_linears, acti="hardswish")

    def forward(self, x):
        x = x.permute(0, 2, 1)           # (B, C, L)
        x = self.DenseNet1D(x)
        x = x.permute(0, 2, 1)           # (B, L, 163)
        x, _ = self.lstm(x)              # (B, L, 192)
        x = x.permute(0, 2, 1)           # (B, 192, L)
        x = self.senet(x)                 # (B, 192, L)
        x = torch.flatten(x, start_dim=1)  # (B, 192 * L)
        return self.mlp(x)

    @staticmethod
    def get_model_params():
        model_params = dict(
            in_channels=config["in_channels"],
            num_linears=config["num_linears"],
            seqlen=config["seqlen"],
            dropout_linears=config["dropout_linears"],
        )
        return model_params

    @staticmethod
    def get_hparams() -> Dict[str, Any]:
        hparams = dict(
            batchsize=config["batchsize"],
            lr=config["lr"],
            patience=config["patience"],
            monitor=config["monitor"],
            name=config["name"],
            max_epochs=config["max_epochs"]
        )
        return hparams