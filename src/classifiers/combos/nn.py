import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from src.callbacks import EarlyStopping
from src.layers import Dense, CombosCorrectionLayer, Input, Dropout
from src.loss_functions.combos import CombosCrossEntropy
from src.metrics.coverage_curves.base import CoverageCurveBase
from src.metrics.top_k_accuracies.top_k_combos_accuracy import TopKCombosAccuracy
from src.models import Model
from src.optimizers import Adam, Adamax, Adagrad, RMSprop, SGD
from src.pipelines.preprocessing.features.models.neural_network import NeuralNetworkFeaturesPreprocessor
from src.transformers.preprocessing.targets.target_corrector import TargetCorrector



class CombosNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 n_layers: int,
                 n_neurons: int,
                 dropout: float=0.0,
                 optimizer_type: str="adam",
                 learning_rate: float=0.001,
                 epochs: int=20,
                 batch_size: int=32,
                 preprocessor: NeuralNetworkFeaturesPreprocessor = None,
                 **kwargs
                 ):
        self.model = None
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer_type = optimizer_type
        self.history = None
        self.forbidden_combos = None
        self.targets_corrector = TargetCorrector()
        self.y_corrector = None
        self.preprocessor = preprocessor
        self.loss_fn = CombosCrossEntropy(**kwargs)
        self._set_optimizer(optimizer_type, learning_rate)
        self.dummy = DummyClassifier(strategy="prior")
        self.early_stopping = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=10,
            restore_best_weights=True
        )

    def _set_optimizer(self, optimizer_type: str, learning_rate: float):
        match optimizer_type:
            case "adam":
                self.optimizer = Adam(learning_rate=learning_rate)
            case "sgd":
                self.optimizer = SGD(learning_rate=learning_rate)
            case "rmsprop":
                self.optimizer = RMSprop(learning_rate=learning_rate)
            case "adagrad":
                self.optimizer = Adagrad(learning_rate=learning_rate)
            case "adamax":
                self.optimizer = Adamax(learning_rate=learning_rate)
            case _:
                raise ValueError(f"Unknown optimizer {optimizer_type}")


    def _build_model(self, input_dim):
        features_input = Input(shape=(input_dim,), name="features_input")
        y_corrector_input = Input(shape=(1326,), name="y_corrector_input")
        x = features_input
        for i in range(self.n_layers):
            x = Dense(self.n_neurons, activation="relu", name=f"hidden_layer_{i+1}")(x)
            if self.dropout:
                x = Dropout(self.dropout, name=f"dropout_{i+1}")(x)
        y_pred = Dense(1326, activation="softmax", name="raw_prediction_layer")(x)
        y_corrected = CombosCorrectionLayer(name="combos_correction_layer")(y_pred, y_corrector_input)
        model_input  = (features_input, y_corrector_input)
        model = Model(
            inputs=model_input,
            outputs=y_corrected,
            name="CombosNNClassifier"
        )
        model.compile(optimizer=self.optimizer,
                      loss=self.loss_fn,
                      metrics=[CoverageCurveBase(), TopKCombosAccuracy(300), ]
                      )
        model.summary()
        return model

    def fit(self, X, y):
        input_dim = X.shape[1]
        X_cards = self.preprocessor.retrieve_known_cards(X)
        y_corrector = self.targets_corrector.fit_transform(X_cards)
        model_inputs = (X, y_corrector)
        self.model = self._build_model(input_dim)
        self.model.fit(
            model_inputs, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            verbose=1,
            callbacks=[self.early_stopping]
        )
        self.history = self.model.history.history
        self.dummy.fit(X, y)
        return self

    def predict_proba(self, X):
        X_cards = self.preprocessor.retrieve_known_cards(X)
        y_corrector = self.targets_corrector.transform(X_cards)
        model_inputs = (X, y_corrector)
        return self.model.predict(model_inputs)

    def dummy_predict_proba(self, X):
        dummy_proba = self.dummy.predict_proba(X)
        print(dummy_proba, type(dummy_proba))
        return dummy_proba

    def predict(self, X):
        predictions = self.predict_proba(X)
        return predictions.argmax(axis=1)

    def dummy_predict(self, X):
        return self.dummy.predict(X)

    def plot_training_graphs(self):
        """
        Plot the training graphs: loss, validation loss, area under coverage curve, validation area under coverage curve
        :return:
        """
        loss_data = self.history["loss"]
        val_loss_data = self.history["val_loss"]
        aucc_data = self.history["area_under_coverage_curve"]
        val_aucc_data = self.history["val_area_under_coverage_curve"]
        epochs = list(range(1, len(loss_data) + 1))
        # Plot training graphs in red and validation graphs in blue
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss", "Area Under Coverage Curve"))
        fig.add_trace(go.Scatter(x=epochs, y=loss_data, mode="lines", name="Training Loss", line=dict(color="red", width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=epochs, y=val_loss_data, mode="lines", name="Validation Loss", line=dict(color="blue", width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=epochs, y=aucc_data, mode="lines", name="Training Aucc", line=dict(color="red", width=2)), row=1, col=2)
        fig.add_trace(go.Scatter(x=epochs, y=val_aucc_data, mode="lines", name="Validation Aucc", line=dict(color="blue", width=2)), row=1, col=2)
        fig.update_layout(title="Training Graphs")
        # Set x-axis title
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        # Set y-axes titles
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Area Under Coverage Curve", row=1, col=2)
        fig.show()


