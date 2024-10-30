from sklearn.pipeline import Pipeline
from src.loaders.raw.shapes import RawShapesLoader
from src.transformers.shapes.category_transformer import CategoryTransformer


class ShapesLoader(Pipeline):
    def __init__(self):
        super().__init__(steps=[
            ("raw_shapes_loader", RawShapesLoader()),
            ("category_transformer", CategoryTransformer())
        ])