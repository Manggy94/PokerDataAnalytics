from sklearn.pipeline import Pipeline
from src.loaders.fixed.cards import CardsLoader
from src.loaders.raw.flops import RawFlopsLoader
from src.pipelines.flops import FlopsPipeline


class FlopsLoader(Pipeline):
    def __init__(self):
        super().__init__(steps=[
            ("raw_flops_loader", RawFlopsLoader()),
            ("flops_pipeline", FlopsPipeline(
                cards=CardsLoader().transform(None)
            ))
        ])