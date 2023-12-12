from sklearn.feature_extraction import text as sk_text

class UnigramTransformer:
    def __init__(self, max_features=30):
        self.vectorizer = sk_text.CountVectorizer(ngram_range=(1, 1), max_features=max_features, stop_words="english")

    def fit(self, text, y=None):
        self.vectorizer.fit(text)

    def transform(self, text, y=None):
        return self.vectorizer.transform(text).toarray()

    def fit_transform(self, text, y=None):
        return self.vectorizer.fit_transform(text).toarray()
