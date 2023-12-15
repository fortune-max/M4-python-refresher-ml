from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from experiments.transformers.task_2_custom_transformer import CustomTransformerTask2

class CustomClassifierTask2:
    def __init__(self, degree=2, random_state=42):
        self.pipeline = Pipeline([
            ('task_2_custom_transformer', CustomTransformerTask2(degree=degree)),
            ('clf', LogisticRegression(max_iter=1000, random_state=random_state, penalty='l2'))
        ])

    def fit(self, X, y=None):
        self.pipeline.fit(X, y)

    def predict(self, X, y=None):
        return self.pipeline.predict(X)
    
    def score(self, X, y=None):
        return self.pipeline.score(X, y)
