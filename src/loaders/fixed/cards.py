from sklearn.pipeline import Pipeline
from src.loaders.raw.cards import RawCardsLoader
from src.loaders.fixed.ranks import RanksLoader
from src.loaders.fixed.suits import SuitsLoader
from src.pipelines.components.cards import CardsPipeline


class CardsLoader(Pipeline):
    def __init__(self):
        super().__init__(steps=[
            ("raw_cards_loader", RawCardsLoader()),
            ("cards_pipeline", CardsPipeline(
                ranks=RanksLoader().transform(None),
                suits=SuitsLoader().transform(None)
            )),
        ])
