import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scikeras.wrappers import KerasClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from src.callbacks import EarlyStopping
from src.layers import Dense, CombosCorrectionLayer, InputLayer, Dropout
from src.loss_functions.combos import CombosCrossEntropy
from src.metrics.coverage_curve import CoverageCurve
from src.metrics.top_k_combos_accuracy import TopKCombosAccuracy
from src.models import Model, Sequential
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

        self.keras_clf = None
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
        self.combos_factor = kwargs.get("combos_factor", 1)
        self.hands_factor = kwargs.get("hands_factor", 1)
        self.suits_factor = kwargs.get("suits_factor", 1)
        self.ranks_factor = kwargs.get("ranks_factor", 1)
        self.first_rank_factor = kwargs.get("first_rank_factor", 1)
        self.first_card_factor = kwargs.get("first_card_factor", 1)
        self.second_rank_factor = kwargs.get("second_rank_factor", 1)
        self.second_card_factor = kwargs.get("second_card_factor", 1)
        self.rank_difference_factor = kwargs.get("rank_difference_factor", 1)
        self.forbidden_combos_factor = kwargs.get("forbidden_combos_factor", 1)
        self.loss_fn = CombosCrossEntropy(
            combos_factor=self.combos_factor,
            hands_factor=self.hands_factor,
            suits_factor=self.suits_factor,
            ranks_factor=self.ranks_factor,
            first_rank_factor=self.first_rank_factor,
            first_card_factor=self.first_card_factor,
            second_rank_factor=self.second_rank_factor,
            second_card_factor=self.second_card_factor,
            rank_difference_factor=self.rank_difference_factor
        )

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

        self.early_stopping = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=10,
            restore_best_weights=True
        )

    def _build_deep_layers(self):
        deep_layers = Sequential(name="Deep Layers")
        for i in range(self.n_layers):
            deep_layers.add(Dense(self.n_neurons, activation="relu", name=f"hidden_layer_{i+1}"))
            if self.dropout:
                deep_layers.add(Dropout(self.dropout))
        return deep_layers


    def _build_model(self, input_dim):
        model = Sequential(name="CombosNNClassifier")
        model.add(InputLayer(shape=(input_dim,), name="input_layer"))
        for i in range(self.n_layers):
            model.add(Dense(self.n_neurons, activation="relu", name=f"hidden_layer_{i+1}"))
            if self.dropout:
                model.add(Dropout(self.dropout))
        model.add(Dense(1326, activation="softmax", name="output_layer"))
        # model.add(CombosCorrectionLayer(y_corrector=self.y_corrector, name="combos_correction_layer"))
        model.compile(optimizer=self.optimizer,
                      loss=self.loss_fn,
                      metrics=[CoverageCurve(), TopKCombosAccuracy(300),]
                      )
        model.summary()
        return model

    def fit(self, X, y):
        input_dim = X.shape[1]
        X_cards = self.preprocessor.retrieve_known_cards(X)
        self.y_corrector = self.targets_corrector.fit_transform(X_cards)
        print(self.y_corrector)
        self.keras_clf = KerasClassifier(
            model=lambda: self._build_model(input_dim),
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            verbose=1,
            callbacks=[self.early_stopping]

        )
        self.keras_clf.fit(X, y)
        self.history = self.keras_clf.model_.history.history
        return self

    def predict(self, X):
        return self.keras_clf.predict(X)

    def predict_proba(self, X):
        return self.keras_clf.predict_proba(X)

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


