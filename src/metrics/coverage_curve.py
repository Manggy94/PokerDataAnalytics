import tensorflow as tf


class CoverageCurve(tf.keras.metrics.Metric):
    """
    Calcule la courbe de couverture pour les k premières étiquettes.
    """
    def __init__(self, name="coverage_curve", **kwargs):
        super(CoverageCurve, self).__init__(name=name, **kwargs)
        self.coverage_curve = self.add_weight(name="coverage_curve", shape=(0,), initializer="zeros", aggregation=tf.VariableAggregation.NONE)
        self.num_labels = self.add_weight(name="num_labels", initializer="zeros", dtype=tf.int32, trainable=False)

    @staticmethod
    def coverage_curve(y_true, y_pred):
        """
        Calcule les valeurs de top-k accuracy pour tous les k de 1 à y_true.shape[1].

        Args:
            y_true (tf.Tensor): Matrice binaire des étiquettes vraies (shape: [n_samples, n_labels]).
            y_pred (tf.Tensor): Matrice des scores prédits (shape: [n_samples, n_labels]).

        Returns:
            coverage_curve (tf.Tensor): Vecteur de précision cumulée pour tous les k.
        """
        # Tri des indices par score décroissant pour chaque échantillon
        sorted_indices = tf.argsort(y_pred, direction="DESCENDING", axis=-1)
        y_true_sorted = tf.gather(y_true, sorted_indices, batch_dims=1)

        # Cumul des étiquettes correctes pour chaque k
        cumulative_correct = tf.cumsum(y_true_sorted, axis=-1)
        total_labels = tf.reduce_sum(y_true, axis=-1, keepdims=True)

        # Calcul des précisions cumulées
        coverage_curve = tf.divide(cumulative_correct, total_labels)  # Diviser par le total de labels positifs
        return coverage_curve

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Calcul de la courbe de couverture pour le batch courant
        batch_coverage_curve = self.coverage_curve(y_true, y_pred)
        self.coverage_curve.assign(tf.reduce_mean(batch_coverage_curve, axis=0))  # Moyenne sur les échantillons du batch
        self.num_labels.assign(y_true.shape[1])

    def result(self):
        return self.coverage_curve[:self.num_labels]

    def reset_states(self):
        self.coverage_curve.assign(tf.zeros((0,), dtype=tf.float32))
        self.num_labels.assign(0)
