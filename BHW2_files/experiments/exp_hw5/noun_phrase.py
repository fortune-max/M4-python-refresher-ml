import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize

class NounPhraseTransformer:
    def __init__(self):
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')

    def get_noun_phrases(self, sentence):
        tagged_sentence = pos_tag(word_tokenize(sentence))
        noun_phrase_pattern = "NP: {<DT>?<JJ>*<NN>}"
        cp = nltk.RegexpParser(noun_phrase_pattern)
        tree = cp.parse(tagged_sentence)
        return [subtree.leaves() for subtree in tree.subtrees() if subtree.label() == 'NP']
    
    def get_noun_phrase_features(self, sentence):
        noun_phrases = self.get_noun_phrases(sentence)
        return " ".join([noun_phrase[0][0] for noun_phrase in noun_phrases])
    
    def fit(self, text, y=None):
        pass

    def transform(self, text, y=None):
        return [self.get_noun_phrase_features(sentence) for sentence in text]
    
    def fit_transform(self, text, y=None):
        return self.transform(text)
