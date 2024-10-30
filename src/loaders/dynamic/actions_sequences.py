from sklearn.pipeline import Pipeline
from src.loaders.raw.actions_sequences import RawActionsSequencesLoader
from src.transformers.category_transformer import CategoryTransformer


class ActionsSequencesLoader(Pipeline):
    def __init__(self):
        super().__init__(steps=[
            ("raw_actions_sequences_loader", RawActionsSequencesLoader()),
            ('category_transformer', CategoryTransformer())
        ])