from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from experiments.transformers.polynomial import PolynomialTransformer

class CustomTransformerTask2:
    def __init__(self, degree=2):
        self.pipeline = Pipeline([
            ('polynomial', PolynomialTransformer(degree=degree)),
            ('scaler', StandardScaler()),
        ])

    def fit(self, X, y=None):
        self.pipeline.fit(X, y)

    def transform(self, X, y=None):
        return self.pipeline.transform(X)
    
    def fit_transform(self, X, y=None):
        return self.pipeline.fit_transform(X, y)
