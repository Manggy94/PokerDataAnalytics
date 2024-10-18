"""A module to perform PCA on a given dataset"""
import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA




class PCAnalyser:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.correlation_matrix = df.corr()
        self.squared_correlation_matrix = self.correlation_matrix ** 2
        self.pca = PCA(random_state=42)
        self.pca.fit(df)
        self.data = self.pca.transform(df)
        self.axis_names = [f'PC{i+1}' for i in range(df.shape[1])]
        self.pca_df = pd.DataFrame(self.data, columns=self.axis_names, index=df.index)
        self.variance_ratios = self.pca.explained_variance_ratio_
        self.cumulative_variance = self.variance_ratios.cumsum()
        self.pca_correlation_table = pd.DataFrame(self.pca.components_, columns=df.columns, index=self.axis_names)




    def display_correlation_matrix(self):
        """Displays a correlation matrix of the given DataFrame"""
        fig = px.imshow(
            self.squared_correlation_matrix,
            aspect='equal',
            color_continuous_scale='magma',
            title='Correlation Matrix',
        )
        fig.update_layout(
            title_x=0.5,
            margin=dict(l=40, r=40, t=40, b=40),
            xaxis_showgrid=False,
            xaxis=dict(showticklabels=False),
            yaxis_showgrid=False,
            yaxis=dict(showticklabels=False),
            coloraxis_colorbar=dict(
                title="Correlation",
                tickvals=[0, 0.5, 1],
                ticktext=["0", "0.5", "1"],
            ),
        )
        fig.show()


    def display_1d_scatter(self):
        fig = px.scatter(
            self.pca_df,
            x='PC1',
            y=[0] * len(self.pca_df),
            title='1D PCA',
            labels={'PC1': 'Principal Component 1'},
            hover_data=['PC1'],
            opacity=0.7
        )
        fig.show()
        print(f"Part of variance explained: {self.cumulative_variance[0]:.2%}")

    def display_2d_scatter(self):

        fig = px.scatter(
            self.pca_df,
            x='PC1',
            y='PC2',
            title='2D PCA',
            labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'},
            hover_data=['PC1', 'PC2'],
            opacity=0.7
        )
        fig.show()
        print(f"Part of variance explained: {self.cumulative_variance[1]:.2%}")

    def display_3d_scatter(self):
        fig = px.scatter_3d(
            self.pca_df,
            x='PC1',
            y='PC2',
            z='PC3',
            title='3D PCA',
            labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2', 'PC3': 'Principal Component 3'},
            hover_data=['PC1', 'PC2', 'PC3'],
            opacity=0.7
        )
        fig.show()
        print(f"Part of variance explained: {self.cumulative_variance[2]:.2%}")

    def display_ratios(self):
        fig = px.bar(
            x=[f'PC{i+1}' for i in range(len(self.variance_ratios))],
            y=self.variance_ratios,
            title='Variance Ratios',
            labels={'x': 'Principal Component', 'y': 'Variance Ratio'},
            opacity=0.7
        )
        fig.show()

    def display_cumulative_variance(self):
        fig = px.line(
            x=[f'PC{i+1}' for i in range(len(self.cumulative_variance))],
            y=self.cumulative_variance,
            title='Cumulative Variance',
            labels={'x': 'Principal Component', 'y': 'Cumulative Variance'},
        )
        # Add a horizontal line at 0.90
        fig.add_hline(y=0.90, line_dash='dash', line_color='red')
        # Add another horizontal line at 0.95
        fig.add_hline(y=0.95, line_dash='dash', line_color='black')
        fig.show()

    def write_pca_report(self):
        # First we want to display the correlation matrix
        self.display_correlation_matrix()
        # Then we want to display the 1D PCA
        print("Displaying 1D PCA")
        self.display_1d_scatter()
        # Then we want to display the 2D PCA
        print("Displaying 2D PCA")
        self.display_2d_scatter()
        # Then we want to display the 3D PCA
        print("Displaying 3D PCA")
        self.display_3d_scatter()
        # Then we want to display the variance ratios
        print("Displaying variance ratios")
        self.display_ratios()
        # Then we want to display the cumulative variance
        print("Displaying cumulative variance")
        self.display_cumulative_variance()


