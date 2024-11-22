import tensorflow as tf


class CoverageCurve(tf.keras.metrics.Metric):
    """
    Calcule la courbe de couverture pour les k premières étiquettes.
    """
    def __init__(self, name="coverage_curve", **kwargs):
        super(CoverageCurve, self).__init__(name=name, **kwargs)
        self.coverage_curve_sum = self.add_weight(name="coverage_curve_sum", shape=(1,), initializer="zeros", dtype=tf.float32)
        self.num_batches = self.add_weight(name="num_batches", initializer="zeros", dtype=tf.int32)

    @staticmethod
    def compute_coverage_curve(y_true, y_pred):
        """
        Calcule les valeurs de top-k accuracy pour tous les k de 1 à y_true.shape[1].

        Args:
            y_true (tf.Tensor): Matrice binaire des étiquettes vraies (shape: [n_samples, n_labels]).
            y_pred (tf.Tensor): Matrice des scores prédits (shape: [n_samples, n_labels]).

        Returns:
            tf.Tensor: Vecteur de précision cumulée pour tous les k.
        """
        # Tri des indices par score décroissant pour chaque échantillon
        sorted_indices = tf.argsort(y_pred, direction="DESCENDING", axis=-1)
        y_true_sorted = tf.gather(y_true, sorted_indices, batch_dims=1)

        # Cumul des étiquettes correctes pour chaque k
        cumulative_correct = tf.cumsum(y_true_sorted, axis=-1)
        total_labels = tf.reduce_sum(y_true, axis=-1, keepdims=True)

        # Calcul des précisions cumulées
        coverage_curve = tf.divide(cumulative_correct, total_labels)  # Diviser par le total de labels positifs
        return tf.reduce_mean(coverage_curve, axis=0)  # Moyenne des courbes sur tous les échantillons du batch

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Met à jour l'état avec les données du batch courant.

        Args:
            y_true (tf.Tensor): Labels vrais (shape: [n_samples, n_labels]).
            y_pred (tf.Tensor): Scores prédits (shape: [n_samples, n_labels]).
            sample_weight (tf.Tensor, optional): Poids des échantillons. Non utilisé ici.
        """
        # Calcul de la courbe de couverture pour le batch courant
        batch_coverage_curve = self.compute_coverage_curve(y_true, y_pred)

        # Ajouter les résultats à l'accumulateur
        self.coverage_curve_sum.assign_add(batch_coverage_curve)
        self.num_batches.assign_add(1)

    def result(self):
        """
        Retourne la couverture moyenne calculée sur tous les batches.

        Returns:
            tf.Tensor: La courbe de couverture moyenne.
        """
        return self.coverage_curve_sum / tf.cast(self.num_batches, tf.float32)

    def reset_states(self):
        """
        Réinitialise les états internes de la métrique.
        """
        self.coverage_curve_sum.assign(tf.zeros_like(self.coverage_curve_sum))
        self.num_batches.assign(0)
