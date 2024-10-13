from sklearn.pipeline import Pipeline
from src.loaders.raw.buy_ins import RawBuyInsLoader
from src.loaders.raw.ref_tournaments import RawRefTournamentsLoader
from src.loaders.raw.tour_speeds import RawTourSpeedsLoader
from src.loaders.raw.tour_types import RawTourTypesLoader
from src.pipelines.ref_tournaments import RefTournamentsPipeline



class RefTournamentsLoader(Pipeline):
    def __init__(self):
        super().__init__(steps=[
            ("raw_ref_tournaments_loader", RawRefTournamentsLoader()),
            ("ref_tournaments_pipeline", RefTournamentsPipeline(
                buy_ins=RawBuyInsLoader().transform(None),
                tour_speeds=RawTourSpeedsLoader().transform(None),
                tour_types=RawTourTypesLoader().transform(None)
            ))
        ])