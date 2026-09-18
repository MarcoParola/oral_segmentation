from .polygons import PolygonDataset


class MultiClassSegmentationDataset(PolygonDataset):
    """Background + category channels; last annotation wins overlapping pixels."""
    def __init__(self, annonations, transform=None, n_classes=3):
        super().__init__(annonations, transform, n_classes=n_classes)
