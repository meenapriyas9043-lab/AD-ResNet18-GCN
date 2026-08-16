"""
backbones.py — Feature extractors / baseline CNN classifiers.

Provides:
  - `get_backbone`: uniform builder for all CNN baselines named in
    config.yaml -> models_compared (googlenet, alexnet, densenet121,
    efficientnet_b0, resnet18_standalone). This is what Reviewer 3
    Comment 4 asked for: real, trained baselines rather than a mention in
    a response letter.
  - `ResNet18FeatureExtractor`: the ResNet18 trunk with its final FC layer
    removed, exposing the 512-d global-average-pooled embedding that feeds
    the graph in graph_utils.py (Reviewer 3 Comment 2).
"""
import torch
import torch.nn as nn
from torchvision import models


class ResNet18FeatureExtractor(nn.Module):
    """ResNet18 with the classification head removed.

    Output: (batch, 512) — the global-average-pooled feature vector used
    as the raw material for BOTH the standalone-ResNet18 ablation model
    and the graph nodes of the hybrid ResNet18+GCN model.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.resnet18(weights=weights)
        self.trunk = nn.Sequential(*list(net.children())[:-1])  # drop the fc layer
        self.out_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.trunk(x)                 # (B, 512, 1, 1)
        return torch.flatten(feats, 1)         # (B, 512)


class ResNet18Standalone(nn.Module):
    """Ablation model: ResNet18 features -> plain FC classifier, NO GCN.

    Used to isolate the contribution of the graph-based relational module
    (Reviewer 3 Comment 4 / the manuscript's ablation claim).
    """

    def __init__(self, n_classes: int, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        self.extractor = ResNet18FeatureExtractor(pretrained=pretrained)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.extractor.out_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extractor(x))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.extractor(x)


def _replace_head(model: nn.Module, in_features: int, n_classes: int, attr_path):
    """Utility to swap a torchvision model's classification head."""
    module = model
    for attr in attr_path[:-1]:
        module = getattr(module, attr)
    setattr(module, attr_path[-1], nn.Linear(in_features, n_classes))
    return model


def get_backbone(name: str, n_classes: int, pretrained: bool = True) -> nn.Module:
    """Uniform builder for every baseline in config.yaml::models_compared.

    Every returned model consumes a (B, 3, 128, 128) tensor and returns
    (B, n_classes) logits, so train.py can treat all baselines identically.
    """
    name = name.lower()

    if name == "googlenet":
        weights = models.GoogLeNet_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.googlenet(weights=weights, aux_logits=True, init_weights=not pretrained)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
        m.aux1.fc2 = nn.Linear(m.aux1.fc2.in_features, n_classes)
        m.aux2.fc2 = nn.Linear(m.aux2.fc2.in_features, n_classes)
        return m

    if name == "alexnet":
        weights = models.AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.alexnet(weights=weights)
        m.classifier[6] = nn.Linear(m.classifier[6].in_features, n_classes)
        return m

    if name == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.densenet121(weights=weights)
        m.classifier = nn.Linear(m.classifier.in_features, n_classes)
        return m

    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.efficientnet_b0(weights=weights)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
        return m

    if name == "resnet18_standalone":
        return ResNet18Standalone(n_classes=n_classes, pretrained=pretrained)

    raise ValueError(f"Unknown backbone '{name}'. See config.yaml::models_compared.")
