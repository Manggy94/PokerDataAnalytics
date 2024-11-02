from sklearn.pipeline import Pipeline

from src.loaders.fixed.tour_speeds import TourSpeedsLoader
from src.loaders.fixed.tour_types import TourTypesLoader
from src.loaders.raw.buy_ins import RawBuyInsLoader
from src.loaders.raw.ref_tournaments import RawRefTournamentsLoader
from src.pipelines.components.ref_tournaments import RefTournamentsPipeline



class RefTournamentsLoader(Pipeline):
    def __init__(self):
        super().__init__(steps=[
            ("raw_ref_tournaments_loader", RawRefTournamentsLoader()),
            ("ref_tournaments_pipeline", RefTournamentsPipeline(
                buy_ins=RawBuyInsLoader().transform(None),
                tour_speeds=TourSpeedsLoader().transform(None),
                tour_types=TourTypesLoader().transform(None)
            ))
        ])