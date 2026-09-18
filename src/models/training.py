"""Device-neutral training metrics shared by the original Lightning models."""
import torch
import torch.nn.functional as F


def segmentation_step(module, batch, stage):
    images, target, _ = batch
    logits = module(images)  # one forward only: no duplicate BatchNorm updates
    labels = target.argmax(1) if module.num_classes > 1 else target
    loss = module.loss(logits, labels)
    with torch.no_grad():
        if module.num_classes == 1:
            predicted = (logits.sigmoid() > module.sgm_threshold).float()
        else:
            predicted = F.one_hot(logits.argmax(1), module.num_classes).permute(0, 3, 1, 2).float()
        accuracy = (predicted.argmax(1) == labels).float().mean() if module.num_classes > 1 else (predicted == target).float().mean()
        p, t = (predicted[:, 1:], target[:, 1:]) if module.num_classes > 1 else (predicted, target)
        dims = (0, 2, 3)
        intersection = (p * t).sum(dims)
        total = (p + t).sum(dims)
        dice = ((2 * intersection + 1e-5) / (total + 1e-5)).mean()
        iou = ((intersection + 1e-5) / (total - intersection + 1e-5)).mean()
    module.log(f"{stage}_loss", loss, on_step=stage == "train", on_epoch=True, batch_size=images.size(0))
    module.log_dict({f"{stage}_acc": accuracy, f"{stage}_dice": dice, f"{stage}_jaccard": iou}, on_step=False, on_epoch=True, batch_size=images.size(0))
    return loss
