from sklearn.pipeline import Pipeline
from src.loaders.raw.cards import RawCardsLoader
from src.loaders.raw.ranks import RawRanksLoader
from src.loaders.raw.suits import RawSuitsLoader
from src.pipelines.cards import CardsPipeline


class CardsLoader(Pipeline):
    def __init__(self):
        super().__init__(steps=[
            ("raw_cards_loader", RawCardsLoader()),
            ("cards_pipeline", CardsPipeline(
                ranks=RawRanksLoader().transform(None),
                suits=RawSuitsLoader().transform(None)
            ))
        ])
