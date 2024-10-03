import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from matplotlib.pyplot import title
from plotly.subplots import make_subplots




class TournamentGraphs:
    def __init__(self, tournaments: pd.DataFrame):
        self.tournaments = tournaments
        self.tournaments_grouped = self.tournaments.groupby(["ref_tournament_buy_in_total", "ref_tournament_type", "ref_tournament_speed", "total_players_range"])
        self.tournaments_resume = self.tournaments_grouped\
            .agg(
                nb_played=("tournament_id", "count"),
                total_profit=("profit", "sum"),
                mean_profit=("profit", "mean"),
                mean_roi=("roi", "mean"),
                mean_finish_percentage=("finish_percentage", "mean"),
                finish_position=("final_position", "mean"),
                total_players=("total_players", "mean"),
                ITM=("ITM", "mean")
            ).reset_index()


    def plot_profit_over_time(self):
        cumulative_profits = pd.DataFrame({
            'start_date': self.tournaments['start_date'],
            'Raw Profit': self.tournaments['profit'].cumsum(),
            'Freeze out Profir': self.tournaments['freeze_out_profit'].cumsum(),
            'Bounty Profit': self.tournaments['bounty_profit'].cumsum(),
        })

        # Tracer le graphique avec les 3 colonnes
        fig = px.line(
            cumulative_profits,
            x='start_date',
            y=['Raw Profit', 'Freeze out Profir', 'Bounty Profit'],
            title='Évolution des profits cumulés au fil du temps',
            labels={'value': 'Profit cumulé (€)', 'start_date': 'Date'},
            color_discrete_map={
                'Raw Profit': 'blue',
                'Freeze out Profir': 'green',
                'Bounty Profit': 'red',
            },

        )

        # Mise en forme de la légende
        fig.update_layout(
            legend_title_text='Type de profit',
            yaxis_title="Cumul des profits (€)",
            xaxis_title="Date",
        )
        fig.show()

    def plot_profit_distribution(self):
        labels = ['Raw Profit', 'Freeze out Profit', 'Bounty Profit']
        data = [self.tournaments['profit'], self.tournaments['freeze_out_profit'], self.tournaments['bounty_profit']]
        fig = ff.create_distplot(data, labels, show_hist=True, show_rug=True)
        fig.update_layout(
            title='Distribution des profits',
            xaxis_title='Profit (€)',
            yaxis_title='Densité',

        )
        fig.show()

    def plot_roi_distribution(self):
        labels = ['ROI', 'Freeze out ROI', 'Bounty ROI']
        tournaments = self.tournaments.dropna(subset=['roi', 'freeze_out_roi', 'bounty_roi', "tournament_type", "speed"])
        data = [tournaments['roi'], tournaments['freeze_out_roi'], tournaments['bounty_roi']]
        fig = ff.create_distplot(data, labels, show_hist=True, show_rug=True)
        fig.update_layout(
            title='Distribution des ROI',
            xaxis_title='ROI',
            yaxis_title='Densité',
            xaxis=dict(range=[-1, 2]),

        )
        fig.show()
        #La figure 2 est un boxplot de la distribution des ROIs pour chaque type de tournoi.
        fig2 = go.Figure()
        fig2.add_trace(go.Box(
            x=tournaments["tournament_type"],
            y=tournaments['roi'],
            name='ROI'))
        fig2.add_trace(go.Box(
            x=tournaments["tournament_type"],
            y=tournaments['freeze_out_roi'],
            name='Freeze out ROI'))
        fig2.add_trace(go.Box(
            x=tournaments["tournament_type"],
            y=tournaments['bounty_roi'],
            name='Bounty ROI'))
        fig2.update_layout(
            title='Distribution des ROI',
            yaxis_title='ROI',
            yaxis=dict(range=[-1, 2]),
        )
        fig2.show()
        # La figure 3 fait un boxplot en fonction de la vitesse des tournois.
        fig3 = go.Figure()
        fig3.add_trace(go.Box(
            x=tournaments["speed"],
            y=tournaments['roi'],
            name='ROI'))
        fig3.add_trace(go.Box(
            x=tournaments["speed"],
            y=tournaments['freeze_out_roi'],
            name='Freeze out ROI'))
        fig3.add_trace(go.Box(
            x=tournaments["speed"],
            y=tournaments['bounty_roi'],
            name='Bounty ROI'))
        fig3.update_layout(
            title='Distribution des ROI',
            yaxis_title='ROI',
            yaxis=dict(range=[-1, 2]),
        )
        fig3.show()

    def plot_roi_vs_finish_position(self):
        # Filtrer les types de tournois désirés
        structured_tournaments = self.tournaments[self.tournaments["ref_tournament_type"].isin(["CLASSIC", "KO", "FLIGHT"])]

        trace1 = go.Scatter(
            x=structured_tournaments["finish_percentage"],
            y=structured_tournaments["freeze_out_roi"],
            mode='markers',
            marker=dict(color='blue', opacity=0.7),
            name='ROI vs Finish Position'
        )
        trace2 = go.Scatter(
            x=structured_tournaments["finish_percentage"],
            y=structured_tournaments["freeze_out_percentage_won"],
            mode='markers',
            marker=dict(color='red', opacity=0.7),
            name='FO Prizepool % won vs Finish Position'
        )

        # Créer une figure
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)

        # Ajouter les points de dispersion
        fig.add_trace(trace1, row=1, col=1)
        fig.add_trace(trace2, row=2, col=1)

        # Mettre à jour la mise en page de la figure
        fig.update_layout(
            title="ROI et % de prizepool gagné en fonction de la position finale",
            xaxis_title="Position finale (%)",
            template="plotly_white",  # Utilisation du thème light
            legend_title="Légende",
            font=dict(size=12),
            hovermode='closest',
            showlegend=True,
            height=800,
            width = 1200
        )

        # Mettre à jour les axes
        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='LightGray', range=[20, 0])
        fig.update_yaxes(
            showgrid=True,
            gridwidth=0.5,
            gridcolor='LightGray',
            range=[-1, structured_tournaments["freeze_out_roi"].max() + 1],
            title="ROI du Freeze-out",
            row=1, col=1
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=0.5,
            gridcolor='LightGray',
            range=[0, 50],
            title="% de prizepool gagné",
            row=2, col=1
        )

        # Afficher la figure
        fig.show()

    def plot_mean_roi_by_categories(self):
        tournaments_resume = self.tournaments_resume[self.tournaments_resume["nb_played"] > 10]
        tournaments_resume = tournaments_resume[tournaments_resume["ref_tournament_buy_in_total"] > 0]
        fig = px.scatter_3d(
            tournaments_resume,
            x='ref_tournament_buy_in_total',
            y='ref_tournament_type',
            z='mean_roi',
            color='total_players_range',
            symbol='ref_tournament_speed',
            title='Mean ROI by categories',
            labels={'ref_tournament_buy_in_total': 'Buy-in total', 'tournament_type': 'Tournament type', 'mean_roi': 'Mean ROI'},
            opacity=0.7,
        )

        fig.update_layout(
            width=800,
            height=800,
        )
        fig.show()

    def plot_mean_profit_by_categories(self):
        tournaments_resume = self.tournaments_resume[self.tournaments_resume["nb_played"] > 10]
        tournaments_resume = tournaments_resume[tournaments_resume["ref_tournament_buy_in_total"] > 0]
        fig = px.scatter_3d(
            tournaments_resume,
            x='ref_tournament_buy_in_total',
            y='ref_tournament_type',
            z='mean_profit',
            color='total_players_range',
            symbol='ref_tournament_speed',
            title='Mean Profit by categories',
            labels={'ref_tournament_buy_in_total': 'Buy-in total', 'ref_tournament_type': 'Tournament type', 'mean_profit': 'Mean Profit'},
            opacity=0.7,
        )
        fig.update_layout(
            width=800,
            height=800,
        )
        fig.show()

    def plot_ITM_by_categories(self):
        tournaments_resume = self.tournaments_resume[self.tournaments_resume["nb_played"] > 10]
        tournaments_resume = tournaments_resume[tournaments_resume["ref_tournament_buy_in_total"] > 0]
        fig = px.scatter_3d(
            tournaments_resume,
            x='ref_tournament_buy_in_total',
            y='ref_tournament_type',
            z='ITM',
            color='total_players_range',
            symbol='ref_tournament_speed',
            title='ITM by categories',
            labels={'ref_tournament_buy_in_total': 'Buy-in total', 'ref_tournament_type': 'Tournament type', 'ITM': 'ITM'},
            opacity=0.7,
        )
        fig.update_layout(
            width=800,
            height=800,
        )
        fig.show()