from .polygons import PolygonDataset


class BinarySegmentationDataset(PolygonDataset):
    """Binary union of every annotation/polygon, with safe empty masks."""
    def __init__(self, annonations, transform=None):
        super().__init__(annonations, transform, n_classes=1)
