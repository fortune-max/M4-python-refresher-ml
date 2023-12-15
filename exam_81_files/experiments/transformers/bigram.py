from sklearn.feature_extraction import text as sk_text

class BigramTransformer:
    def __init__(self, max_features=30, min_df=0.01, max_df=0.95):
        self.vectorizer = sk_text.CountVectorizer(ngram_range=(2, 2), max_features=max_features, stop_words="english", min_df=min_df, max_df=max_df)

    def fit(self, text, y=None):
        self.vectorizer.fit(text)

    def transform(self, text, y=None):
        return self.vectorizer.transform(text).toarray()

    def fit_transform(self, text, y=None):
        return self.vectorizer.fit_transform(text).toarray()
