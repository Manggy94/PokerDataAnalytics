import tensorflow as tf


class AreaUnderCoverageCurve(tf.keras.metrics.Metric):
    def __init__(self, name="area_under_coverage_curve", **kwargs):
        super(AreaUnderCoverageCurve, self).__init__(name=name, **kwargs)
        self.aucc_accumulator = self.add_weight(name="aucc", initializer="zeros", dtype=tf.float32)
        self.num_batches = self.add_weight(name="num_batches", initializer="zeros", dtype=tf.int32)

    @staticmethod
    def compute_area_under_coverage_curve(y_true, y_pred):
        """
        Calcule l'aire sous la Coverage Curve (AUCC).

        Args:
            y_true (tf.Tensor): Matrice binaire des étiquettes vraies, shape = [n_samples, n_labels].
            y_pred (tf.Tensor): Matrice des scores prédits, shape = [n_samples, n_labels].

        Returns:
            tf.Tensor: AUCC pour l'ensemble des données.
        """
        # Étape 1 : Tri des étiquettes selon les scores prédits
        sorted_indices = tf.argsort(y_pred, direction="DESCENDING", axis=-1)
        y_true_sorted = tf.gather(y_true, sorted_indices, batch_dims=1)

        # Étape 2 : Calcul de la Coverage Curve
        cumulative_correct = tf.cumsum(y_true_sorted, axis=-1)
        total_labels = tf.reduce_sum(y_true, axis=-1, keepdims=True)
        coverage_curve = tf.divide(cumulative_correct, total_labels)

        # Étape 3 : Calcul de l'aire sous la courbe (moyenne cumulative)
        aucc = tf.reduce_mean(tf.reduce_sum(coverage_curve, axis=-1) / tf.cast(tf.shape(y_true)[1], tf.float32))

        return aucc

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Calculer l'AUCC pour le batch courant
        batch_aucc = self.compute_area_under_coverage_curve(y_true, y_pred)

        # Mettre à jour les accumulateurs
        self.aucc_accumulator.assign_add(batch_aucc)
        self.num_batches.assign_add(1)

    def result(self):
        # Retourner la moyenne des AUCC sur tous les batches
        return self.aucc_accumulator / tf.cast(self.num_batches, tf.float32)

    def reset_states(self):
        # Réinitialiser les accumulateurs
        self.aucc_accumulator.assign(0.0)
        self.num_batches.assign(0)
