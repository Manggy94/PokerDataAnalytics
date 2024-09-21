import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff



class TournamentGraphs:
    def __init__(self, tournaments: pd.DataFrame):
        self.tournaments = tournaments

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
