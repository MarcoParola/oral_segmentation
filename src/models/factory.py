"""Factory for the project's actual Lightning FCN/DeepLab/U-Net classes."""
from torch import nn


def build_model(name="fcn", classes=1, weights=None, encoder_name="resnet50", **kwargs):
    kwargs.update(classes=classes, loss=nn.BCEWithLogitsLoss() if classes == 1 else nn.CrossEntropyLoss())
    if name == "fcn":
        from .fcn import FcnSegmentationNet
        return FcnSegmentationNet(weights=weights, **kwargs)
    if name == "deeplab":
        from .deeplab import DeeplabSegmentationNet
        return DeeplabSegmentationNet(weights=weights, **kwargs)
    if name == "unet":
        from .unet import unetSegmentationNet
        return unetSegmentationNet(encoder_name=encoder_name, encoder_weights="imagenet" if weights else None, **kwargs)
    raise ValueError(f"Unknown model {name}")
