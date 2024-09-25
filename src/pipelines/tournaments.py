import pandas as pd
from sklearn.pipeline import Pipeline
from src.transformers.tournaments.final_position_imputer import FinalPositionImputer
from src.transformers.tournaments.id_typer import IdTyper
from src.transformers.tournaments.tournaments_ref_tournaments_merger import TournamentRefTournamentsMerger


class TournamentsPipeline(Pipeline):
    def __init__(self, ref_tournaments: pd.DataFrame):
        self.ref_tournaments = ref_tournaments
        super().__init__(steps=[
            ("final_position_imputer", FinalPositionImputer()),
            ("id_typer", IdTyper()),
            ("tournament_ref_tournaments_merger", TournamentRefTournamentsMerger(ref_tournaments=ref_tournaments)),
        ])
