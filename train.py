"""Original Hydra + Lightning training entry (CPU/CUDA).

For explicit single-NPU FP32 eager bring-up use ``python -m scripts.smoke``;
Lightning 2.1 does not natively register an NPU accelerator.
"""
from pathlib import Path
import hydra
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader
from torchvision import transforms as T

from src.datasets import BinarySegmentationDataset, MultiClassSegmentationDataset
from src.devices import resolve_device
from src.models.factory import build_model


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg):
    spec = cfg.train.device
    if spec.startswith("npu"):
        raise RuntimeError("Lightning 2.1 NPU integration is not implemented. Use python -m scripts.smoke --device npu:0 for FP32 eager bring-up; no fallback.")
    device = resolve_device(spec)
    pl.seed_everything(None if cfg.train.seed == -1 else cfg.train.seed, workers=True)
    transform = T.Compose([T.Resize((cfg.dataset.resize, cfg.dataset.resize)), T.ToTensor()])
    dataset_cls = BinarySegmentationDataset if cfg.model.num_classes == 1 else MultiClassSegmentationDataset
    extra = {} if cfg.model.num_classes == 1 else {"n_classes": cfg.model.num_classes}
    train_dataset = dataset_cls(hydra.utils.to_absolute_path(cfg.dataset.train), transform=transform, **extra)
    val_dataset = dataset_cls(hydra.utils.to_absolute_path(cfg.dataset.val), transform=transform, **extra)
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers)
    classes = 1 if cfg.model.num_classes == 1 else cfg.model.num_classes + 1
    model = build_model(cfg.model.model_type, classes, weights="DEFAULT" if cfg.train.pretrained else None,
                        encoder_name=cfg.model.encoder_name, lr=cfg.train.lr, epochs=cfg.train.max_epochs,
                        sgm_type=cfg.model.sgm_type, sgm_threshold=cfg.model.sgm_threshold,
                        len_dataset=len(train_dataset), batch_size=cfg.train.batch_size, max_lr=cfg.train.max_lr)
    loggers = []
    if cfg.log.tensorboard:
        from pytorch_lightning.loggers import TensorBoardLogger
        loggers.append(TensorBoardLogger(cfg.log.path, name="oral"))
    if cfg.log.wandb:
        raise ValueError("Remote medical-data logging is disabled in bring-up; use local TensorBoard")
    save_path = Path(hydra.utils.to_absolute_path(cfg.train.save_path))
    checkpoint = ModelCheckpoint(dirpath=save_path, filename="{epoch}-{step}", save_last=True, monitor="val_loss", mode="min")
    trainer = pl.Trainer(
        logger=loggers or False,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=cfg.train.patience), checkpoint],
        accelerator="gpu" if device.type == "cuda" else "cpu",
        devices=[device.index] if device.type == "cuda" else 1,
        precision="32-true", log_every_n_steps=1,
        max_epochs=cfg.train.max_epochs, max_steps=cfg.train.max_steps,
        limit_train_batches=cfg.train.limit_train_batches,
        limit_val_batches=cfg.train.limit_val_batches,
        num_sanity_val_steps=0,
        default_root_dir=str(save_path),
    )
    trainer.fit(model, train_loader, val_loader)
    print(f"Last checkpoint: {checkpoint.last_model_path}")


if __name__ == "__main__":
    main()
