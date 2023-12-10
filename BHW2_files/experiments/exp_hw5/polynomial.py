from sklearn.preprocessing import PolynomialFeatures

class PolynomialTransformer:
    def __init__(self, degree):
        self.vectorizer = PolynomialFeatures(degree=degree)

    def fit(self, X):
        self.vectorizer.fit(X)

    def transform(self, X):
        return self.vectorizer.transform(X)

    def fit_transform(self, X):
        return self.vectorizer.fit_transform(X)
