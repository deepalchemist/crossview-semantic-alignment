

from mmf.common.registry import registry
from .dataset import MovingFashionDataset
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder


@registry.register_builder("movingfashion")
class MovingFashionBuilder(MMFDatasetBuilder):
    def __init__(
        self, dataset_name="movingfashion", dataset_class=MovingFashionDataset, *args, **kwargs
    ):
        super().__init__(dataset_name, dataset_class, *args, **kwargs)

    @classmethod
    def config_path(cls):
        return "configs/datasets/movingfashion/defaults.yaml"
