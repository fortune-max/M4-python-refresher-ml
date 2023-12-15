from sklearn.base import BaseEstimator, RegressorMixin
from experiments.transformers.bigram import BigramTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from experiments.transformers.polynomial import PolynomialTransformer
from xgboost import XGBClassifier

from sklearn.linear_model import Lasso

class CustomClassifierTask4(BaseEstimator, RegressorMixin):
    
    def __init__(self, degree=2, alpha=0.081, random_state=42, n_components=2):
        self.degree = degree
        self.alpha = alpha
        self.random_state = random_state
        self.n_components = n_components

        self.pipeline = Pipeline([
            ('polynomial', PolynomialTransformer(degree=self.degree)),
            ('pca', PCA(n_components=self.n_components)),
            ('clf', Lasso(random_state=self.random_state, alpha=self.alpha))
        ])

    def fit(self, X, y=None):
        self.pipeline.fit(X, y)

    def predict(self, X, y=None):
        return self.pipeline.predict(X)
    
    def score(self, X, y=None):
        return self.pipeline.score(X, y)
    
    def get_params(self, deep=True):
        return {
            'degree': self.degree,
            'alpha': self.alpha,
            'random_state': self.random_state,
            'n_components': self.n_components
        }
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
