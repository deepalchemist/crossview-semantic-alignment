

# Last Change:  2024-04-28 11:33:48

from mmf.common.registry import registry
from .dataset import VideoToShopDataset
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder


@registry.register_builder("videotoshop")
class VideoToShopBuilder(MMFDatasetBuilder):
    def __init__(
        self, dataset_name="videotoshop", dataset_class=VideoToShopDataset, *args, **kwargs
    ):
        super().__init__(dataset_name, dataset_class, *args, **kwargs)

    @classmethod
    def config_path(cls):
        return "configs/datasets/videotoshop/defaults.yaml"
