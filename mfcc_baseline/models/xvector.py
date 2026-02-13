import torch
import torch.nn as nn
import torch.nn.functional as F


class TDNNLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        context_size: int,
        dilation: int,
    ):
        super().__init__()
        self.context_size = context_size
        self.dilation = dilation
        self.conv = nn.Conv1d(
            input_dim,
            output_dim,
            kernel_size=context_size,
            dilation=dilation,
        )
        self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)


class StatsPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=2)
        std = x.std(dim=2)
        return torch.cat([mean, std], dim=1)


class XVectorModel(nn.Module):
    def __init__(
        self,
        input_size: int = 40,
        embedding_size: int = 512,
        num_classes: int = None,
        channel_sizes: list = None,
    ):
        super().__init__()
        
        if channel_sizes is None:
            channel_sizes = [512, 512, 512, 512, 1500]
        
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        
        self.frame1 = TDNNLayer(input_size, channel_sizes[0], context_size=5, dilation=1)
        self.frame2 = TDNNLayer(channel_sizes[0], channel_sizes[1], context_size=3, dilation=2)
        self.frame3 = TDNNLayer(channel_sizes[1], channel_sizes[2], context_size=3, dilation=3)
        self.frame4 = TDNNLayer(channel_sizes[2], channel_sizes[3], context_size=1, dilation=1)
        self.frame5 = TDNNLayer(channel_sizes[3], channel_sizes[4], context_size=1, dilation=1)
        
        self.stats_pooling = StatsPooling()
        
        fc_input_dim = channel_sizes[4] * 2
        self.segment6 = nn.Linear(fc_input_dim, embedding_size)
        self.bn_segment6 = nn.BatchNorm1d(embedding_size)
        
        self.segment7 = nn.Linear(embedding_size, embedding_size)
        self.bn_segment7 = nn.BatchNorm1d(embedding_size)
        
        if num_classes is not None:
            self.fc_out = nn.Linear(embedding_size, num_classes)
        else:
            self.fc_out = None

    def forward(self, x: torch.Tensor, return_embedding: bool = True) -> torch.Tensor:
        x = self.frame1(x)
        x = self.frame2(x)
        x = self.frame3(x)
        x = self.frame4(x)
        x = self.frame5(x)
        
        x = self.stats_pooling(x)
        
        x = self.segment6(x)
        x = self.bn_segment6(x)
        x = F.relu(x)
        
        if return_embedding:
            return x
        
        x = self.segment7(x)
        x = self.bn_segment7(x)
        x = F.relu(x)
        
        if self.fc_out is not None:
            x = self.fc_out(x)
        
        return x

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x, return_embedding=True)


class XVectorClassifier(nn.Module):
    def __init__(
        self,
        input_size: int = 40,
        embedding_size: int = 512,
        num_classes: int = 3,
    ):
        super().__init__()
        self.encoder = XVectorModel(
            input_size=input_size,
            embedding_size=embedding_size,
            num_classes=None,
        )
        self.classifier = nn.Linear(embedding_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.encoder(x, return_embedding=True)
        return self.classifier(embedding)


def test_xvector():
    model = XVectorModel(input_size=40, embedding_size=512, num_classes=10)
    x = torch.randn(2, 40, 100)
    out = model(x, return_embedding=True)
    print(f"Embedding shape: {out.shape}")
    assert out.shape == (2, 512)
    
    out = model(x, return_embedding=False)
    print(f"Output shape (with classification): {out.shape}")
    assert out.shape == (2, 10)


if __name__ == "__main__":
    test_xvector()
