import pandas as pd
from sklearn.pipeline import Pipeline

from src.transformers.utils.category_transformer import CategoryTransformer
from src.transformers.components.ref_tournaments.column_names_corrector import ColumnNamesCorrector
from src.transformers.components.ref_tournaments.ref_tournaments_buy_ins_merger import RefTournamentsBuyInsMerger
from src.transformers.components.ref_tournaments.ref_tournaments_tour_speeds_merger import RefTournamentsTourSpeedsMerger
from src.transformers.components.ref_tournaments.ref_tournaments_tour_types_merger import RefTournamentsTourTypesMerger


class RefTournamentsPipeline(Pipeline):
    def __init__(
            self,
            buy_ins: pd.DataFrame,
            tour_speeds: pd.DataFrame,
            tour_types: pd.DataFrame
    ):
        self.buy_ins = buy_ins
        self.tour_speeds = tour_speeds
        self.tour_types = tour_types
        super().__init__(steps=[
            ("category_transformer", CategoryTransformer()),
            ("ref_tournaments_buy_ins_merger", RefTournamentsBuyInsMerger(buy_ins=buy_ins)),
            ("ref_tournaments_tour_speeds_merger", RefTournamentsTourSpeedsMerger(tour_speeds=tour_speeds)),
            ("ref_tournaments_tour_types_merger", RefTournamentsTourTypesMerger(tour_types=tour_types)),
            ("column_names_corrector", ColumnNamesCorrector())
        ])