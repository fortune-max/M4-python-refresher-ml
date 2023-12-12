from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from ..exp_hw5.polynomial import PolynomialTransformer
from xgboost import XGBClassifier


class PolynomialClassifierPipeline:
    def __init__(self, n_components=2, degree=3, random_state=42):
        print("PolynomialClassifierPipeline initialized with PCA", n_components, "components and Degree", degree)
        self.pipeline = Pipeline([
            ('polynomial', PolynomialTransformer(degree=degree)),
            ('pca', PCA(n_components=n_components)),
            ('clf', LogisticRegression(max_iter=1000, random_state=random_state, penalty='l2'))
        ])

    def fit(self, text, y):
        self.pipeline.fit(text, y)

    def predict(self, text):
        return self.pipeline.predict(text)

    def predict_proba(self, text):
        return self.pipeline.predict_proba(text)

    def score(self, text, y):
        return self.pipeline.score(text, y)


class PolynomialXGBClassifierPipeline:
    def __init__(self, n_components=2, degree=3, random_state=42):
        print("PolynomialXGBClassifierPipeline initialized with PCA", n_components, "components and Degree", degree)
        self.pipeline = Pipeline([
            ('polynomial', PolynomialTransformer(degree=degree)),
            ('pca', PCA(n_components=n_components)),
            ('clf', XGBClassifier(random_state=random_state))
        ])

    def fit(self, text, y):
        self.pipeline.fit(text, y)

    def predict(self, text):
        return self.pipeline.predict(text)

    def predict_proba(self, text):
        return self.pipeline.predict_proba(text)

    def score(self, text, y):
        return self.pipeline.score(text, y)
