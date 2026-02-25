"""
This module implements the FOVNet architecture for learning on boundary representation (B-Rep) 3D models. 

Classes:
    FOVNet: Core network architecture
    FOVNetModule: PyTorch Lightning module for training/evaluation
"""

import torch
import torch.nn.functional as F
from torch import nn
import lightning.pytorch as pl
import torchmetrics
import fovnet.encoders


# ============================================================================
# Classification Head
# ============================================================================

class _NonLinearClassifier(nn.Module):
    """Three-layer MLP classification head with batch normalization and dropout.
    
    Args:
        input_dim: Dimension of input features
        num_classes: Number of output classes
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(self, input_dim, num_classes, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 256, bias=False)
        self.bn1 = nn.BatchNorm1d(256)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(256, 128, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(128, num_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        """Initialize layer weights using Kaiming uniform initialization."""
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, inp):
        """Forward pass through classification head.
        
        Args:
            inp: Input features (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Class logits (batch_size, num_classes)
        """
        x = F.relu(self.bn1(self.linear1(inp)))
        x = self.dp1(x)
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)
        return x


# ============================================================================
# FOVNet Architecture
# ============================================================================

class FOVNet(nn.Module):
    """
    
    Combines multiple geometric representations of B-Rep models:
    - Vision grids from ray-casting (outer/inner hemispheres)
    - UV-grid surface parametrizations
    - Raw face geometric features
    
    Supports both whole-model classification and per-face segmentation.
    """

    def __init__(
        self,
        num_classes,
        vision=False,
        vision_az=12,
        vision_el=6,
        ov_channels=[0, 1, 3],
        iv_channels=[2, 3, 4],
        crv_emb_dim=128,
        srf_emb_dim=128,
        vision_emb_dim=128,
        graph_emb_dim=128,
        dropout=0.3,
        local_uv=False,
        input_graph_dim=64,
        segmentation=False,
        use_face_feat=False,
        face_feat_emb_dim=64,
        use_uv = True,
        use_ov=True,
        use_iv=True,
    ):
        """
        Initialize the UV-Net solid classification model
        
        Args:
            num_classes (int): Number of classes to output
            crv_emb_dim (int, optional): Embedding dimension for the 1D edge UV-grids. Defaults to 64.
            srf_emb_dim (int, optional): Embedding dimension for the 2D face UV-grids. Defaults to 64.
            graph_emb_dim (int, optional): Embedding dimension for the graph. Defaults to 128.
            dropout (float, optional): Dropout for the final non-linear classifier. Defaults to 0.3.
            use_face_feat (bool, optional): If True, use face features in addition to other features. Defaults to False.
            face_feat_emb_dim (int, optional): Embedding dimension for face features. Defaults to 64.
            use_ov (bool, optional): If True, use outer vision (upper hemisphere). Defaults to True.
            use_iv (bool, optional): If True, use inner vision (lower hemisphere). Defaults to True.
        """
        super().__init__()
        self.vision = vision
        self.vision_emb_dim = vision_emb_dim
        self.ov_channels = ov_channels
        self.iv_channels = iv_channels
        self.srf_emb_dim = srf_emb_dim
        self.crv_emb_dim = crv_emb_dim
        self.uv_features = "x_local" if local_uv else "x"
        self.input_graph_dim = input_graph_dim
        self.segmentation = segmentation
        self.use_face_feat = use_face_feat
        self.face_feat_emb_dim = face_feat_emb_dim
        self.use_uv = use_uv
        self.use_ov = use_ov
        self.use_iv = use_iv

        # Helper functions
        def build_vision_encoder(channels, output_dim=None):
            if output_dim is None:
                output_dim = self.vision_emb_dim
            return fovnet.encoders.VisionGridEncoder(
                input_az=vision_az, input_el=vision_el, in_channels=len(channels), output_dims=output_dim)

        def build_surface_encoder(output_dim=None):
            if output_dim is None:
                output_dim = self.srf_emb_dim
            return fovnet.encoders.SurfaceEncoder(in_channels=7, output_dims=output_dim)

        # Initialize encoders
        self.ov_encoder, self.iv_encoder = None, None
        self.surf_encoder = None
        feature_dims = []

        if vision:
            # Only upper/lower hemisphere vision - controlled by use_ov and use_iv
            if use_ov:
                self.ov_encoder = build_vision_encoder(ov_channels)
                feature_dims += [self.vision_emb_dim]
            if use_iv:
                self.iv_encoder = build_vision_encoder(iv_channels)
                feature_dims += [self.vision_emb_dim]

        if use_uv:
            self.surf_encoder = build_surface_encoder()
            feature_dims += [self.srf_emb_dim]

        if use_face_feat:
            feature_dims += [7]

        # Shared FC fusion layer
        self.total_raw_features = sum(feature_dims)
        self.shared_fc = fovnet.encoders.combination_fc(self.total_raw_features, input_graph_dim, bias=False)

        self.graph_encoder = fovnet.encoders.GraphEncoder(input_graph_dim, graph_emb_dim)

        # Classifiers
        self.clf = _NonLinearClassifier(graph_emb_dim, num_classes, dropout)
        self.seg = _NonLinearClassifier(2*graph_emb_dim, num_classes, dropout)
        
    def forward(self, batched_graph):
        """
        Forward pass

        Args:
            batched_graph (dgl.Graph): A batched DGL graph containing the face 2D UV-grids in node features
                                       (ndata['x']) and 1D edge UV-grids in the edge features (edata['x']).

        Returns:
            torch.tensor: Logits (batch_size x num_classes)
        """
        # Preprocessing: Permute dimensions for model compatibility
        if self.use_uv:
            # Permute UV features from (N, H, W, C) to (N, C, H, W) for Conv2D
            for key in ["x", "x_local"]:
                if key in batched_graph.ndata:
                    batched_graph.ndata[key] = batched_graph.ndata[key].permute(0, 3, 1, 2).contiguous()
        
        if self.vision and "vision_grids" in batched_graph.ndata:
            # Permute vision grids from (N, H, W, C) to (N, C, H, W) for Conv2D
            batched_graph.ndata["vision_grids"] = batched_graph.ndata["vision_grids"].permute(0, 3, 1, 2).contiguous()
        
        # Remove unnecessary node features
        required_node_keys = {self.uv_features, "vision_grids"}
        if self.use_face_feat:
            required_node_keys.add("face_feat")
        for key in list(batched_graph.ndata.keys()):
            if key not in required_node_keys:
                batched_graph.ndata.pop(key)

        # Remove unnecessary edge features
        required_edge_keys = set()  # No edge features needed for this model
        for key in list(batched_graph.edata.keys()):
            if key not in required_edge_keys:
                batched_graph.edata.pop(key)

        # Surface features
        surface_feat = batched_graph.ndata.get(self.uv_features, None)

        # Vision features: select only relevant channels
        vision_feat = batched_graph.ndata.get("vision_grids", None)
        
        # Face features: first 7 channels if available
        face_feat = batched_graph.ndata.get("face_feat", None)
        if face_feat is not None and self.use_face_feat:
            face_feat = face_feat[:, :7]  # Take only first 7 channels

        # --- Encode features ---
        encoded_feats = []

        if self.use_uv and surface_feat is not None:
            encoded_feats.append(self.surf_encoder(surface_feat))

        if self.vision and vision_feat is not None:
            # ov_feat: outer vision (upper hemisphere) - only if use_ov is True
            if self.use_ov and self.ov_encoder is not None:
                ov_feat = self.ov_encoder(vision_feat[:, self.ov_channels, :, :])
                encoded_feats.append(ov_feat)
            # iv_feat: inner vision (lower hemisphere) - only if use_iv is True
            if self.use_iv and self.iv_encoder is not None:
                iv_feat = self.iv_encoder(vision_feat[:, self.iv_channels, :, :])
                encoded_feats.append(iv_feat)
            
        if self.use_face_feat and face_feat is not None:
            encoded_feats.append(face_feat)

        # Concatenate raw outputs → shared FC (or use dummy ones when no features present)
        if self.total_raw_features > 0:
            cat_feats = torch.cat(encoded_feats, dim=1)
            hidden_feat = self.shared_fc(cat_feats)
        else:
            # No raw features: use a learnable shared vector
            num_nodes = batched_graph.number_of_nodes()
            hidden_feat = batched_graph.in_degrees().float().unsqueeze(1)  # (N, 1)
        # --- Graph encoding ---
        node_emb, graph_emb = self.graph_encoder(batched_graph, hidden_feat)

        # --- Segmentation or classification ---
        if self.segmentation:
            num_nodes = batched_graph.batch_num_nodes().to(graph_emb.device)
            graph_emb_expanded = graph_emb.repeat_interleave(num_nodes, dim=0)
            out = self.seg(torch.cat((node_emb, graph_emb_expanded), dim=1))
        else:
            out = self.clf(graph_emb)

        return out

class FOVNetModule(pl.LightningModule):
    """PyTorch Lightning module for FOVNet training and evaluation."""

    def __init__(self, **kwargs):
        """
        Args:
            lr (float): Learning rate for the optimizer.
            **kwargs: Arguments passed to the FOVNet model, including `num_classes`.
        """
        # Pop non-model arguments before saving hyperparameters
        kwargs.pop('weights_only', None)
        
        super().__init__()
        self.save_hyperparameters()
        
        # Separate model kwargs from LightningModule kwargs
        model_kwargs = self.hparams.copy()
        model_kwargs.pop('lr', None)
        model_kwargs.pop('train_random_rotation', None)
        self.model = FOVNet(**model_kwargs)

        # Initialize metrics for different stages
        self.metrics = nn.ModuleDict()
        num_classes = self.hparams.num_classes
        for stage in ["train", "val", "test"]:
            stage_metrics = nn.ModuleDict()
            metric_args = {"task": "multiclass", "num_classes": num_classes}
            stage_metrics["acc"] = torchmetrics.Accuracy(**metric_args)
            if self.hparams.get("segmentation", False):
                stage_metrics["iou"] = torchmetrics.JaccardIndex(**metric_args)
            self.metrics[f"{stage}_metrics"] = stage_metrics

    def forward(self, batched_graph):
        """Forward pass through the model."""
        return self.model(batched_graph)

    def _shared_step(self, batch, stage):
        """Shared logic for training, validation, and testing steps."""
        inputs = batch["graph"].to(self.device)
        labels = (
            inputs.ndata["y"] if self.hparams.get("segmentation", False) else batch["label"]
        ).to(self.device).long()
        
        logits = self(inputs)
        loss = F.cross_entropy(logits, labels)
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Update metrics
        preds = torch.argmax(logits, dim=1) if not self.hparams.get("segmentation", False) else torch.softmax(logits, dim=-1)
        for metric_name, metric in self.metrics[f"{stage}_metrics"].items():
            metric(preds, labels)
            self.log(f"{stage}_{metric_name}", metric, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        """Configure the optimizer."""
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)