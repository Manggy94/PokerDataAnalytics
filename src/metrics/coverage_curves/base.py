import numpy as np
import plotly.graph_objects as go
import tensorflow as tf
from sklearn.dummy import DummyClassifier


class CoverageCurveBase(tf.keras.metrics.Metric):
    """
    Metric that computes the Area Under Coverage Curve (AUCC) for multi-label classification tasks.
    """
    def __init__(self, name="area_under_coverage_curve", **kwargs):
        super(CoverageCurveBase, self).__init__(name=name, **kwargs)
        self.aucc_accumulator = self.add_weight(name="aucc", initializer="zeros", dtype=tf.float32)
        self.num_batches = self.add_weight(name="num_batches", initializer="zeros", dtype=tf.int32)
        self.dummy = DummyClassifier(strategy="prior")

    @staticmethod
    def compute_coverage_curve(y_true, y_pred):
        """
        Computes the coverage curve vector.

        Args:
            y_true (tf.Tensor): Binary matrix of true labels (shape: [n_samples, n_labels]).
            y_pred (tf.Tensor): Float matrix of predicted scores (shape: [n_samples, n_labels]).

        Returns:
            tf.Tensor: The coverage curve vector.
        """
        sorted_indices = tf.argsort(y_pred, direction="DESCENDING", axis=-1)
        y_true_sorted = tf.gather(y_true, sorted_indices, batch_dims=1)
        cumulative_correct = tf.cumsum(y_true_sorted, axis=-1)
        total_labels = tf.reduce_sum(y_true, axis=-1, keepdims=True)

        # Calcul des précisions cumulées
        coverage_curve = tf.divide(cumulative_correct, total_labels)  # Diviser par le total de labels positifs
        return tf.reduce_mean(coverage_curve, axis=0)  # Moyenne des courbes sur tous les échantillons du batch

    def compute_area_under_coverage_curve(self, y_true, y_pred):
        """
        Calcule l'aire sous la courbe de couverture pour les k premières étiquettes.

        Args:
            y_true (tf.Tensor): Matrice binaire des étiquettes vraies (shape: [n_samples, n_labels]).
            y_pred (tf.Tensor): Matrice des scores prédits (shape: [n_samples, n_labels]).

        Returns:
            tf.Tensor: L'aire sous la courbe de couverture.
        """
        coverage_curve = self.compute_coverage_curve(y_true, y_pred)
        aucc = tf.reduce_mean(coverage_curve)
        return aucc

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Updates Area Under Coverage Curve metric.

        Args:
            y_true (tf.Tensor): Labels vrais (shape: [n_samples, n_labels]).
            y_pred (tf.Tensor): Scores prédits (shape: [n_samples, n_labels]).
            sample_weight (tf.Tensor, optional): Poids des échantillons. Non utilisé ici.
        """
        # Calcul de la courbe de couverture pour le batch courant
        batch_aucc = self.compute_area_under_coverage_curve(y_true, y_pred)
        self.aucc_accumulator.assign_add(batch_aucc)
        self.num_batches.assign_add(1)

    def result(self):
        """
        Retourne la couverture moyenne calculée sur tous les batches.

        Returns:
            tf.Tensor: La courbe de couverture moyenne.
        """
        return self.aucc_accumulator / tf.cast(self.num_batches, tf.float32)

    def reset_states(self):
        """
        Réinitialise les états internes de la métrique.
        """
        self.aucc_accumulator.assign(0.0)
        self.num_batches.assign(0)

    def plot_coverage_curve(self, y_true, y_pred, dummy_proba, target="Combos"):
        """
        Trace la courbe de couverture à partir de y_true et y_pred.
        """
        coverage_vector = self.compute_coverage_curve(y_true, y_pred).numpy()
        dummy_coverage_vector = self.compute_coverage_curve(y_true, dummy_proba).numpy()

        aucc = self.compute_area_under_coverage_curve(y_true, y_pred)
        dummy_auc = self.compute_area_under_coverage_curve(y_true, dummy_proba)
        relative_diff = (aucc - dummy_auc) / dummy_auc
        x_labels = np.arange(1, len(coverage_vector) + 1) / len(coverage_vector) * 100
        x_ticks = np.linspace(0, 100, 9)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=coverage_vector,
                mode="lines+markers",
                name=f"{target} Model Coverage Curve",
                marker=dict(size=6, color="blue"),
                line=dict(color="blue", width=2),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=dummy_coverage_vector,
                mode="lines+markers",
                name="Dummy Coverage Curve",
                marker=dict(size=6, color="red"),
                line=dict(color="red", width=2),
            )
        )

        # Mise en page
        fig.update_layout(
            title=f"{target} Coverage Curve (AUCC: {aucc:.4f} vs Dummy AUCC: {dummy_auc:.4f}), Relative Diff: {relative_diff:.2%}",
            xaxis=dict(
                title="Coverage (%)",
                tickmode="array",
                tickvals=x_ticks,
                ticktext=[f"{t:.1f}%" for t in x_ticks],
            ),
            yaxis=dict(
                title="Coverage Value",
            ),
            template="plotly_white",
        )
        fig.update_yaxes(range=[0, 1])


        fig.show()
