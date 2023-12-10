from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from ..exp_hw5.unigram import UnigramTransformer


class UnigramClassifierPipeline:
    def __init__(self, n_components=2):
        self.pipeline = Pipeline([
            ('unigram', UnigramTransformer()),
            ('pca', PCA(n_components=n_components)),
            ('clf', LogisticRegression(max_iter=1000))
        ])

    def fit(self, text, y):
        self.pipeline.fit(text, y)

    def predict(self, text):
        return self.pipeline.predict(text)

    def predict_proba(self, text):
        return self.pipeline.predict_proba(text)

    def score(self, text, y):
        return self.pipeline.score(text, y)
