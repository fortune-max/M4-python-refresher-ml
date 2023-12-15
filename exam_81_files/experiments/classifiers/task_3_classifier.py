from experiments.transformers.bigram import BigramTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from xgboost import XGBClassifier

class CustomClassifierTask3:
    def __init__(self, max_features=30, random_state=42, n_components=2, min_df=0.01, max_df=0.95):
        self.pipeline = Pipeline([
            ('bigram', BigramTransformer(max_features=max_features, min_df=min_df, max_df=max_df)),
            ('pca', PCA(n_components=n_components)),
            ('clf', XGBClassifier(random_state=random_state))
        ])

    def fit(self, X, y=None):
        self.pipeline.fit(X, y)

    def predict(self, X, y=None):
        return self.pipeline.predict(X)
    
    def score(self, X, y=None):
        return self.pipeline.score(X, y)
