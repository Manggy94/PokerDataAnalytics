import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ProfitsCalculator(BaseEstimator, TransformerMixin):

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X['total_won'] = X['amount_won'] + X['bounty_won']
        X["ITM"] = X["amount_won"] > 0
        X['total_investment'] = X['nb_entries'] * X['ref_tournament_buy_in_total']
        X['classic_investment'] = ((X['ref_tournament_buy_in_prize_pool_contribution'] + X['ref_tournament_buy_in_rake']) *
                                             X['nb_entries'])
        X["bounty_investment"] = X["nb_entries"] * X["ref_tournament_buy_in_bounty"]
        X["prize_pool_percentage_won"] = X["total_won"] / X["prize_pool"]*100
        X["freeze_out_percentage_won"] = X["amount_won"] / X["prize_pool"]*100
        X["profit"] = X["total_won"] - X["total_investment"]
        X["freeze_out_profit"] = X["amount_won"] - X["classic_investment"]
        X["bounty_profit"] = X["bounty_won"] - X["bounty_investment"]
        X['roi'] = X["profit"] / X['total_investment']
        X["freeze_out_roi"] = X["freeze_out_profit"] / X["classic_investment"]
        X["bounty_roi"] = X["bounty_profit"] / X["bounty_investment"]
        X['roi'] = X['roi'].replace([np.inf, -np.inf], np.nan)
        X["freeze_out_roi"] = X["freeze_out_roi"].replace([np.inf, -np.inf], np.nan)
        X["bounty_roi"] = X["bounty_roi"].replace([np.inf, -np.inf], np.nan)
        X["finish_percentage"] = X["final_position"] / X["total_players"]*100
        return X
