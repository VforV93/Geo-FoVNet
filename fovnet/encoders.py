"""Feature encoder modules for FOVNet.

Provides CNN and GNN encoders for vision grids, LRF-UV grids, and graph processing.
"""

import torch
from torch import nn
import torch.nn.functional as F
from dgl.nn.pytorch.glob import MaxPooling
from dgl.nn.pytorch import GATConv


def _conv2d(in_channels, out_channels, kernel_size, padding=0, bias=False, DP=0):
    """
    Helper function to create a 2D convolutional layer with batchnorm and LeakyReLU activation

    Args:
        in_channels (int): Input channels
        out_channels (int): Output channels
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
        padding (int, optional): Padding size on each side. Defaults to 0.
        bias (bool, optional): Whether bias is used. Defaults to False.

    Returns:
        nn.Sequential: Sequential contained the Conv2d, BatchNorm2d and LeakyReLU layers
    """
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        ),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(),
        nn.Dropout(DP),
    )

def _fc(in_features, out_features, bias=False):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=bias),
        nn.BatchNorm1d(out_features),
        nn.LeakyReLU(),
        nn.Dropout(0.1)
    )

def combination_fc(in_features, out_features, bias=False):
    return nn.Sequential(
        nn.Linear(in_features, 256, bias=bias),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, out_features, bias=bias),
        nn.BatchNorm1d(out_features),
        nn.ReLU(),
        nn.Dropout(0.1)
    )

class SurfaceEncoder(nn.Module):
    def __init__(
        self,
        in_channels=7,
        output_dims=64
    ):
        """
        This is the 2D convolutional network that extracts features from the B-rep face
        geometry described as 2D UV-grids (see Section 3.2, Curve & surface convolution
        in paper)

        Args:
            in_channels (int, optional): Number of channels in the edge UV-grids. By default
                                         we expect 3 channels for point coordinates and 3 for
                                         surface normals and 1 for the trimming mask. Defaults
                                         to 7.
            output_dims (int, optional): Output surface embedding dimension. Defaults to 64.
        """
        super(SurfaceEncoder, self).__init__()
        self.in_channels = in_channels
        self.conv1 = _conv2d(in_channels, 32, 3, padding=1, bias=False)
        self.conv2 = _conv2d(32, 64, 3, padding=1, bias=False)
        self.conv3 = _conv2d(64, 128, 3, padding=1, bias=False)
        self.final_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = _fc(128, output_dims, bias=False)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        assert x.size(1) == self.in_channels
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.final_pool(x)
        x = x.view(batch_size, -1)
        
        return self.fc(x)
          
class VisionGridEncoder(nn.Module):
    """
    Encoder for hemisphere vision grids with minimal global pattern detection.
    """
    def __init__(self, input_az=12, input_el=6, in_channels=3, output_dims=128):
        super().__init__()
        self.input_az = input_az
        self.input_el = input_el
        self.in_channels = in_channels
        # Add 2 for coordinate channels
        self.conv1 = nn.Conv2d(in_channels + 2, 32, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1))
        self.fc = _fc(64, output_dims)
    
    def forward(self, x):
        """
        Args:
            x: (batch, channels, elevation, azimuth)
        """
        batch_size = x.size(0)
        device = x.device
        # Create coordinate channels
        el_range = torch.linspace(0, 1, steps=self.input_el, device=device).view(1, 1, self.input_el, 1).expand(batch_size, 1, self.input_el, self.input_az)
        az_range = torch.linspace(0, 1, steps=self.input_az, device=device).view(1, 1, 1, self.input_az).expand(batch_size, 1, self.input_el, self.input_az)
        x = torch.cat([x, el_range, az_range], dim=1)  # (batch, channels+2, el, az)

        def conv_circular(x, conv, bn):
            x = F.pad(x, (1, 1, 0, 0), mode='circular')
            x = F.pad(x, (0, 0, 1, 1), mode='constant', value=0)
            x = F.relu(bn(conv(x)))
            return x

        # Local features (your existing CNN)
        x = conv_circular(x, self.conv1, self.bn1)
        x = conv_circular(x, self.conv2, self.bn2)
        x = self.pool(x).flatten(1)

        return self.fc(x)

class GraphEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=64,
        num_layers=3,
        num_heads=4,
        feat_drop=0,
        attn_drop=0,
        residual=True,
    ):
        super(GraphEncoder, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.gat_layers = nn.ModuleList()

        # First GAT layer
        self.gat_layers.append(
            GATConv(
                in_feats=input_dim,
                out_feats=hidden_dim,
                num_heads=num_heads,
                feat_drop=feat_drop,
                attn_drop=attn_drop,
                residual=residual,
                activation=F.elu,
                allow_zero_in_degree = True
            )
        )

        # Hidden GAT layers
        for _ in range(1, num_layers - 1):
            self.gat_layers.append(
                GATConv(
                    in_feats=hidden_dim * num_heads,
                    out_feats=hidden_dim,
                    num_heads=num_heads,
                    feat_drop=feat_drop,
                    attn_drop=attn_drop,
                    residual=residual,
                    activation=F.elu,
                    allow_zero_in_degree = True
                )
            )

        # Output projection layer
        self.gat_layers.append(
            GATConv(
                in_feats=hidden_dim * num_heads,
                out_feats=output_dim,
                num_heads=1,
                feat_drop=feat_drop,
                attn_drop=attn_drop,
                residual=False,
                activation=None,
                allow_zero_in_degree = True
            )
        )

        self.pool = MaxPooling()

    def forward(self, g, h):
        hidden_rep = [h]

        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h)
            # GATConv returns shape (N, num_heads, out_feats)
            h = h.flatten(1) if h.dim() == 3 else h
            hidden_rep.append(h)

        out = hidden_rep[-1]

        # Global pooling for graph-level representation
        pooled = self.pool(g, out)
        return out, pooled