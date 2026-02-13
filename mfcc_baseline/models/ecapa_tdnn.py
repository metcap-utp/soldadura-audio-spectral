import torch
import torch.nn as nn
import torch.nn.functional as F


class SEModule(nn.Module):
    def __init__(self, channels: int, bottleneck: int = 128):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)


class Res2NetBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int = 3,
        scale: int = 8,
        dilation: int = 1,
    ):
        super().__init__()
        self.scale = scale
        self.width = output_channels // scale
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for i in range(scale - 1):
            self.convs.append(
                nn.Conv1d(
                    self.width,
                    self.width,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=dilation * (kernel_size - 1) // 2,
                )
            )
            self.bns.append(nn.BatchNorm1d(self.width))
        
        self.conv_input = nn.Conv1d(input_channels, output_channels, kernel_size=1)
        self.bn_input = nn.BatchNorm1d(output_channels)
        self.se = SEModule(output_channels, bottleneck=128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_input(x)
        x = self.bn_input(x)
        
        spx = torch.split(x, self.width, dim=1)
        outputs = [spx[0]]
        
        for i in range(1, self.scale):
            if i == 1:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i - 1](sp)
            sp = self.bns[i - 1](sp)
            sp = F.relu(sp)
            outputs.append(sp)
        
        x = torch.cat(outputs, dim=1)
        x = self.se(x)
        
        return x


class TDNNBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=dilation * (kernel_size - 1) // 2,
        )
        self.bn = nn.BatchNorm1d(output_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, channels: int, attention_channels: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(channels * 2, attention_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(attention_channels, channels, kernel_size=1),
            nn.Softmax(dim=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.size(2)
        global_x = torch.cat(
            [
                x.mean(dim=2, keepdim=True).expand_as(x),
                x.std(dim=2, keepdim=True).expand_as(x),
            ],
            dim=1,
        )
        w = self.attention(global_x)
        
        mean = torch.sum(x * w, dim=2)
        std = torch.sqrt(torch.sum((x ** 2) * w, dim=2) - mean ** 2 + 1e-5)
        
        return torch.cat([mean, std], dim=1)


class ECAPA_TDNN(nn.Module):
    def __init__(
        self,
        input_size: int = 40,
        channels: list = None,
        lin_neurons: int = 192,
        activation: nn.Module = nn.ReLU,
        kernel_sizes: list = None,
        dilations: list = None,
        attention_channels: int = 128,
        res2net_scale: int = 8,
        se_channels: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        if channels is None:
            channels = [512, 512, 512, 512, 1536]
        if kernel_sizes is None:
            kernel_sizes = [5, 3, 3, 3, 1]
        if dilations is None:
            dilations = [1, 2, 3, 4, 1]
        
        self.input_size = input_size
        self.channels = channels
        
        self.layer1 = TDNNBlock(
            input_size, channels[0], kernel_sizes[0], dilations[0], dropout
        )
        
        self.res2net_blocks = nn.ModuleList()
        for i in range(1, len(channels) - 1):
            self.res2net_blocks.append(
                Res2NetBlock(
                    channels[i - 1],
                    channels[i],
                    kernel_size=kernel_sizes[i],
                    scale=res2net_scale,
                    dilation=dilations[i],
                )
            )
        
        self.mfa_conv = TDNNBlock(
            channels[-2] * (len(channels) - 2),
            channels[-1],
            kernel_sizes[-1],
            dilations[-1],
            dropout,
        )
        
        self.asp = AttentiveStatisticsPooling(channels[-1], attention_channels)
        self.asp_bn = nn.BatchNorm1d(channels[-1] * 2)
        
        self.fc = nn.Conv1d(channels[-1] * 2, lin_neurons, kernel_size=1)
        self.bn = nn.BatchNorm1d(lin_neurons)

    def forward(self, x: torch.Tensor, return_embedding: bool = True) -> torch.Tensor:
        x1 = self.layer1(x)
        
        x2_features = []
        for res2net in self.res2net_blocks:
            x2_features.append(res2net(x1))
        
        x2 = torch.cat(x2_features, dim=1)
        
        x_mfa = self.mfa_conv(x2)
        
        x_asp = self.asp(x_mfa)
        x_asp = x_asp.unsqueeze(2)
        x_asp = self.asp_bn(x_asp)
        x_asp = F.relu(x_asp)
        
        x = self.fc(x_asp)
        x = x.squeeze(2)
        x = self.bn(x)
        
        if return_embedding:
            return x
        
        return x

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x, return_embedding=True)


class ECAPA_TDNNClassifier(nn.Module):
    def __init__(
        self,
        input_size: int = 40,
        num_classes: int = 3,
        channels: list = None,
        lin_neurons: int = 192,
    ):
        super().__init__()
        self.encoder = ECAPA_TDNN(
            input_size=input_size,
            channels=channels,
            lin_neurons=lin_neurons,
        )
        self.classifier = nn.Linear(lin_neurons, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.encoder(x, return_embedding=True)
        return self.classifier(embedding)


def test_ecapa():
    model = ECAPA_TDNN(input_size=40, lin_neurons=192)
    x = torch.randn(2, 40, 100)
    out = model(x, return_embedding=True)
    print(f"Embedding shape: {out.shape}")
    assert out.shape == (2, 192)
    
    classifier = ECAPA_TDNNClassifier(input_size=40, num_classes=3)
    out = classifier(x)
    print(f"Classification output shape: {out.shape}")
    assert out.shape == (2, 3)


if __name__ == "__main__":
    test_ecapa()
