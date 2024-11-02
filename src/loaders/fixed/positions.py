from sklearn.pipeline import Pipeline
from src.loaders.raw.positions import RawPositionsLoader
from src.pipelines.components.positions import PositionsPipeline


class PositionsLoader(Pipeline):
    def __init__(self):
        super().__init__(steps=[
            ("raw_positions_loader", RawPositionsLoader()),
            ("positions_pipeline", PositionsPipeline())
        ])