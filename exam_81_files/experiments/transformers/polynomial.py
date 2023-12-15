from sklearn.preprocessing import PolynomialFeatures

class PolynomialTransformer:
    def __init__(self, degree):
        self.vectorizer = PolynomialFeatures(degree=degree)

    def fit(self, X, y=None):
        self.vectorizer.fit(X)

    def transform(self, X, y=None):
        return self.vectorizer.transform(X)

    def fit_transform(self, X, y=None):
        return self.vectorizer.fit_transform(X)
