# src/preprocess.py
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

stopset = set(stopwords.words("english"))
stemmer = PorterStemmer()

def normalize_text(text, do_stem=True):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stopset and len(t) > 1]
    if do_stem:
        tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)

# Example usage
if __name__=="__main__":
    s = "Tokenization, stemming & stop-words! Are handled well."
    print(normalize_text(s))
