import pandas as pd
from sklearn.pipeline import Pipeline

from src.transformers.utils.category_transformer import CategoryTransformer
from src.transformers.utils.float_converter import FloatConverter as GlobalFloatConverter
from src.transformers.components.tournaments.start_date_type_corrector import StartDateTypeCorrector
from src.transformers.components.tournaments.final_position_imputer import FinalPositionImputer
from src.transformers.components.tournaments.id_typer import IdTyper
from src.transformers.components.tournaments.float_converter import FloatConverter
from src.transformers.components.tournaments.int_converter import IntConverter
from src.transformers.components.tournaments.profits_calculator import ProfitsCalculator
from src.transformers.components.tournaments.total_players_splitter import TotalPlayersSplitter
from src.transformers.components.tournaments.tournaments_ref_tournaments_merger import TournamentRefTournamentsMerger

class TournamentsPipeline(Pipeline):
    def __init__(self, ref_tournaments: pd.DataFrame):
        self.ref_tournaments = ref_tournaments
        super().__init__(steps=[
            ("final_position_imputer", FinalPositionImputer()),
            ("id_typer", IdTyper()),
            ("tournament_ref_tournaments_merger", TournamentRefTournamentsMerger(ref_tournaments=ref_tournaments)),
            ("int_converter", IntConverter()),
            ("date_type_corrector", StartDateTypeCorrector()),
            ("total_players_splitter", TotalPlayersSplitter()),
            ("profits_calculator", ProfitsCalculator()),
            ("float_converter", FloatConverter()),
            ("category_transformer", CategoryTransformer()),
            ("global_float_converter", GlobalFloatConverter())


        ])
