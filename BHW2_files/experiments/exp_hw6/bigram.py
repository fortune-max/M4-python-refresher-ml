from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from ..exp_hw5.bigram import BigramTransformer
from xgboost import XGBClassifier

class BigramClassifierPipeline:
    def __init__(self, n_components=2, max_features=30, random_state=42):
        print("BigramClassifierPipeline initialized with PCA", n_components, "components and CountVectorizer", max_features, "max features")
        self.pipeline = Pipeline([
            ('bigram', BigramTransformer(max_features=max_features)),
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

class BigramXGBClassifierPipeline:
    def __init__(self, n_components=2, max_features=30, random_state=42):
        print("BigramXGBClassifierPipeline initialized with PCA", n_components, "components and CountVectorizer", max_features, "max features")
        self.pipeline = Pipeline([
            ('bigram', BigramTransformer(max_features=max_features)),
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
