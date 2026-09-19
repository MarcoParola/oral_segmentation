import json
import socket
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms as T

from src.datasets import BinarySegmentationDataset, MultiClassSegmentationDataset
from src.devices import resolve_device
from src.models.factory import build_model
from src.models.training import segmentation_step


@pytest.fixture
def dataset_json(tmp_path):
    (tmp_path / "oral1").mkdir()
    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(tmp_path / "oral1" / "sample.png")
    data = {"images": [{"id": 1, "file_name": "sample.png", "width": 16, "height": 16},
                       {"id": 2, "file_name": "sample.png", "width": 16, "height": 16}],
            "categories": [{"id": i} for i in (1, 2, 3)],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1,
                 "segmentation": [[1,1,4,1,4,4,1,4], [6,1,8,1,8,3,6,3]]},
                {"id": 2, "image_id": 1, "category_id": 3,
                 "segmentation": [[10,10,14,10,14,14,10,14]]}]}
    path = tmp_path / "train.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_binary_all_polygons_and_empty_mask(dataset_json):
    dataset = BinarySegmentationDataset(dataset_json)
    image, mask, category = dataset[0]
    assert image.shape == (3,16,16) and mask.shape == (1,16,16)
    assert mask[0,2,2] == mask[0,2,7] == mask[0,12,12] == 1
    assert category == -1
    _, empty, category = dataset[1]
    assert torch.isfinite(empty).all() and empty.sum() == 0 and category == 0


def test_nested_archive_layout(dataset_json):
    root = dataset_json.parent / "oral1"
    (root / "oral1").mkdir()
    (root / "sample.png").rename(root / "oral1" / "sample.png")
    assert BinarySegmentationDataset(dataset_json)[0][0].shape == (3, 16, 16)


def test_multiclass_nearest_and_background(dataset_json):
    dataset = MultiClassSegmentationDataset(dataset_json, transform=T.Compose([T.Resize((31,31)), T.ToTensor()]))
    _, mask, _ = dataset[0]
    assert mask.shape == (4,31,31)
    assert set(mask.unique().tolist()) == {0.0,1.0}
    assert torch.all(mask.sum(0) == 1)
    assert mask[1].sum() > 0 and mask[3].sum() > 0 and mask[2].sum() == 0


def test_random_unpaired_transform_rejected(dataset_json):
    dataset = BinarySegmentationDataset(dataset_json, transform=T.RandomHorizontalFlip())
    with pytest.raises(ValueError, match="unpaired"):
        dataset[0]


def test_invalid_polygon_rejected(dataset_json):
    data = json.loads(dataset_json.read_text())
    data["annotations"][0]["segmentation"] = [[1,2,3]]
    dataset_json.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="XY"):
        BinarySegmentationDataset(dataset_json)[0]


def test_device_selection_no_fallback():
    with patch("src.devices.importlib.import_module", side_effect=AssertionError("Unexpected NPU import")):
        assert resolve_device("cpu").type == "cpu"
    with patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError, match="no fallback"):
            resolve_device("cuda:0")
    with patch("src.devices.importlib.import_module", side_effect=ImportError("No torch_npu")):
        with pytest.raises(RuntimeError, match="no fallback"):
            resolve_device("npu:0")
    with patch("src.devices.importlib.import_module"), patch.object(torch, "npu", SimpleNamespace(is_available=lambda: False), create=True):
        with pytest.raises(RuntimeError, match="no fallback"):
            resolve_device("npu:0")
    with pytest.raises(ValueError):
        resolve_device("auto")


def test_single_forward_metrics():
    class Dummy(torch.nn.Module):
        num_classes = 1
        sgm_threshold = 0.5
        loss = torch.nn.BCEWithLogitsLoss()
        calls = 0
        def forward(self, x):
            self.calls += 1
            return x
        def log(self, name, value, **kwargs):
            pass
        def log_dict(self, values, **kwargs):
            self.values = values
    model = Dummy()
    loss = segmentation_step(model, (torch.tensor([[[[-10.,10.]]]]), torch.tensor([[[[0.,1.]]]]), torch.tensor([1])), "val")
    assert model.calls == 1 and loss.isfinite()
    assert model.values["val_dice"] == 1 and model.values["val_acc"] == 1


@pytest.mark.parametrize("name", ["fcn", "deeplab", "unet"])
def test_original_model_offline_forward_and_scheduler(name):
    torch.set_num_threads(2)
    with patch.object(socket.socket, "connect", side_effect=AssertionError("Network disabled")), \
         patch("torch.hub.download_url_to_file", side_effect=AssertionError("Weight download disabled")):
        model = build_model(name, weights=None, encoder_name="resnet50", len_dataset=4, batch_size=2, epochs=2)
        model.eval()
        with torch.no_grad():
            logits = model(torch.zeros(1,3,32,32))
        assert logits.shape == (1,1,32,32) and torch.isfinite(logits).all()
        config = model.configure_optimizers()
        assert config["lr_scheduler"]["interval"] == "step"
        cls = type(model)
        del config, model
        multiclass = cls(classes=4, **({"encoder_name": "resnet50"} if name == "unet" else {}))
        assert isinstance(multiclass.loss, torch.nn.CrossEntropyLoss)
        multiclass.all_preds.append(1)
        multiclass.all_labels.append(1)
        multiclass.on_test_epoch_start()
        assert not multiclass.all_preds and not multiclass.all_labels
