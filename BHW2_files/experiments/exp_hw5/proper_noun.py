import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize

class ProperNounTransformer:
    def __init__(self):
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')

    def get_proper_nouns(self, sentence):
        tagged_sentence = pos_tag(word_tokenize(sentence))
        proper_noun_pattern = "NP: {<NNP>}"
        cp = nltk.RegexpParser(proper_noun_pattern)
        tree = cp.parse(tagged_sentence)
        return [subtree.leaves() for subtree in tree.subtrees() if subtree.label() == 'NP']
    
    def get_proper_noun_features(self, sentence):
        proper_nouns = self.get_proper_nouns(sentence)
        return " ".join([proper_noun[0][0] for proper_noun in proper_nouns])
    
    def fit(self, text, y=None):
        pass

    def transform(self, text, y=None):
        return [self.get_proper_noun_features(sentence) for sentence in text]

    def fit_transform(self, text, y=None):
        return self.transform(text)
