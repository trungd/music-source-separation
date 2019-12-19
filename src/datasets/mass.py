from dlex import MainConfig
from dlex.datasets import DatasetBuilder


class MASS(DatasetBuilder):
    def __init__(self, params: MainConfig):
        super().__init__(
            params,
            downloads=["https://zenodo.org/record/1300286#.WzYMMhyxXMU"])

