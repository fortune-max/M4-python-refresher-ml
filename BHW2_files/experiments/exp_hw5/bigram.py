from sklearn.feature_extraction import text as sk_text

class BigramTransformer:
    def __init__(self):
        self.vectorizer = sk_text.CountVectorizer(ngram_range=(2, 2), max_features=100, stop_words="english")

    def fit(self, text):
        self.vectorizer.fit(text)

    def transform(self, text):
        return self.vectorizer.transform(text)

    def fit_transform(self, text):
        return self.vectorizer.fit_transform(text)