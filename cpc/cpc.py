import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

URLS = {
    "cpc": "https://github.com/bshall/cpc/releases/download/v0.1/cpc-d7475380.pt",
    "kmeans50": "https://github.com/bshall/cpc/releases/download/v0.1/kmeans50-89accca9.pt",
    "kmeans100": "https://github.com/bshall/cpc/releases/download/v0.1/kmeans100-c7eda98e.pt",
}


class ChannelNorm(nn.Module):
    def __init__(self, channels, epsilon=1e-05):
        super().__init__()
        self.weight = nn.parameter.Parameter(torch.Tensor(1, channels, 1))
        self.bias = nn.parameter.Parameter(torch.Tensor(1, channels, 1))
        self.epsilon = epsilon

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True)
        x = (x - mean) * torch.rsqrt(var + self.epsilon)
        x = x * self.weight + self.bias
        return x


class Encoder(nn.Module):
    def __init__(self, channels=512):
        super().__init__()

        self.conv0 = nn.Conv1d(1, channels, 10, stride=5, padding=3)
        self.norm0 = ChannelNorm(channels)
        self.conv1 = nn.Conv1d(channels, channels, 8, stride=4, padding=2)
        self.norm1 = ChannelNorm(channels)
        self.conv2 = nn.Conv1d(channels, channels, 4, stride=2, padding=1)
        self.norm2 = ChannelNorm(channels)
        self.conv3 = nn.Conv1d(channels, channels, 4, stride=2, padding=1)
        self.norm3 = ChannelNorm(channels)
        self.conv4 = nn.Conv1d(channels, channels, 4, stride=2, padding=1)
        self.norm4 = ChannelNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.norm0(self.conv0(x)))
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.relu(self.norm3(self.conv3(x)))
        x = F.relu(self.norm4(self.conv4(x)))
        return x


class Context(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(512, 512, batch_first=True, num_layers=2)
        self.h = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm(x)
        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x, h = self.lstm(x, self.h)
        self.h = h
        return x


class CPC(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.context = Context()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x).transpose(1, 2)
        c = self.context(x)
        return c, x

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x).transpose(1, 2)
        c = self.context.encode(x)
        return c


def cpc(pretrained: bool = True, progress: bool = True) -> CPC:
    r"""
    CPC-big model from https://arxiv.org/abs/2011.11588.
    Args:
        pretrained (bool): load pretrained weights into the model
        progress (bool): show progress bar when downloading model
    """
    cpc = CPC()
    if pretrained:
        cpc_state = torch.hub.load_state_dict_from_url(URLS["cpc"], progress=progress)
        cpc.load_state_dict(cpc_state)
        cpc.eval()
    return cpc


def _kmeans(
    num_clusters: int, pretrained: bool = True, progress: bool = True
) -> KMeans:
    r"""
    k-means checkpoint from https://arxiv.org/abs/2108.00917.
    Args:
        num_clusters (int): number of clusters
        pretrained (bool): load pretrained weights into the model
        progress (bool): show progress bar when downloading model
    """
    kmeans = KMeans(num_clusters)
    if pretrained:
        kmeans_state = torch.hub.load_state_dict_from_url(
            URLS[f"kmeans{num_clusters}"], progress=progress
        )
        kmeans.__dict__["n_features_in_"] = kmeans_state["n_features_in_"]
        kmeans.__dict__["_n_threads"] = kmeans_state["_n_threads"]
        kmeans.__dict__["cluster_centers_"] = kmeans_state["cluster_centers_"].numpy()
    return kmeans


def kmeans50(pretrained: bool = True, progress: bool = True) -> KMeans:
    r"""
    k-means checkpoint from https://arxiv.org/abs/2108.00917 with 50 clusters.
    Args:
        pretrained (bool): load pretrained weights into the model
        progress (bool): show progress bar when downloading model
    """
    return _kmeans(50, pretrained, progress)


def kmeans100(pretrained: bool = True, progress: bool = True) -> KMeans:
    r"""
    k-means checkpoint from https://arxiv.org/abs/2108.00917 with 100 clusters.
    Args:
        pretrained (bool): load pretrained weights into the model
        progress (bool): show progress bar when downloading model
    """
    return _kmeans(100, pretrained, progress)
