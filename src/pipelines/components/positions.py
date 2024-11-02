from sklearn.pipeline import Pipeline

from src.transformers.utils.category_transformer import CategoryTransformer
from src.transformers.components.positions.columns_cleaner import PositionsColumnsCleaner


class PositionsPipeline(Pipeline):
    def __init__(self):
        super().__init__(steps=[
            ('category_transformer', CategoryTransformer()),
            ("positions_cleaner", PositionsColumnsCleaner())
        ])