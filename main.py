import pandas as pd

train_set_raw = pd.read_csv("./data/train.csv", index_col='id')
test_set_raw = pd.read_csv("./data/test.csv", index_col='id')

import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

shadow_vectorizer = CountVectorizer(
    tokenizer=nltk.word_tokenize,
    stop_words=stopwords.words('english'),
    ngram_range=(1, 3)
)


def lem_analyzer_builder(base_analyzer):
    stemmer = nltk.stem.WordNetLemmatizer()

    def analyzer(doc):
        return [stemmer.lemmatize(word.lower()) for word in base_analyzer(doc)]

    return analyzer


vectorizer = CountVectorizer(
    analyzer=lem_analyzer_builder(shadow_vectorizer.build_analyzer()),
    max_features=2000,
    ngram_range=(1, 3))

import numpy as np

np.save('./data/out/test_feed.npy', vectorizer.fit_transform(test_set_raw['review']).toarray())

