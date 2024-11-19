import tensorflow as tf


class FractionAmplifiedLoss(tf.keras.losses.Loss):
    def __init__(self, name="fraction_amplified_loss"):
        super(FractionAmplifiedLoss, self).__init__(name=name)

    def call(self, y_true, y_pred):
        """
        Fonction de perte basée sur la somme des x / (1 - x),
        où x est l'erreur absolue entre y_true et y_pred.

        Args:
            y_true: Tensor des valeurs vraies.
            y_pred: Tensor des valeurs prédites (sorties softmax).

        Returns:
            Une valeur scalaire correspondant à la perte totale.
        """
        # Calcul de l'erreur absolue
        error = tf.abs(y_true - y_pred)

        # Calcul de la perte: x / (1 - x)
        loss = error / (1 - error + 1e-7)
        # Somme sur toutes les dimensions
        total_loss = tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
        return total_loss