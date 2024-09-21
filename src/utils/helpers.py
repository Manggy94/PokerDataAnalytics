import os
import pandas as pd
from config.settings import ANALYTICS_DATA_DIR


def export_full_tournaments():
    print("Exporting full tournaments to CSV file...")
    tournaments = pd.read_csv(f'{ANALYTICS_DATA_DIR}/tournaments.csv', index_col=0)
    ref_tournaments = pd.read_csv(f'{ANALYTICS_DATA_DIR}/ref_tournaments.csv', index_col=0)
    tour_types = pd.read_csv(f'{ANALYTICS_DATA_DIR}/tour_types.csv', index_col=0)
    tour_speeds = pd.read_csv(f'{ANALYTICS_DATA_DIR}/tour_speeds.csv', index_col=0)
    buy_ins = pd.read_csv(f'{ANALYTICS_DATA_DIR}/buy_ins.csv', index_col=0)
    full_tournaments = tournaments.merge(ref_tournaments, left_on='ref_tournament', right_on='id', how='left',
                                         suffixes=('', '_ref'), validate="m:1") \
        .merge(buy_ins, left_on='buy_in', right_on='id', how='left', suffixes=('', '_buy_in'), validate="m:1") \
        .merge(tour_types, left_on='tournament_type', right_on='id', how='left', suffixes=('', '_tour_type'),
               validate="m:1") \
        .merge(tour_speeds, left_on='speed', right_on='id', how='left', suffixes=('', '_tour_speed'),
               validate="m:1")
    destination_path = os.path.join(ANALYTICS_DATA_DIR, 'full_tournaments.csv')
    full_tournaments.to_csv(destination_path, index=True)
    print(f"All full tournaments exported to {destination_path}")
