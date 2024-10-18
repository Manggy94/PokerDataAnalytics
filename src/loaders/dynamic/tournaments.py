from sklearn.pipeline import Pipeline
from src.loaders.raw.tournaments import RawTournamentsLoader
from src.loaders.dynamic.ref_tournaments import RefTournamentsLoader
from src.pipelines.tournaments import TournamentsPipeline


class TournamentsLoader(Pipeline):
    def __init__(self):
        super().__init__(steps=[
            ("raw_tournaments_loader", RawTournamentsLoader()),
            ("tournaments_pipeline", TournamentsPipeline(
                ref_tournaments=RefTournamentsLoader().fit_transform(None)
            ))

        ])