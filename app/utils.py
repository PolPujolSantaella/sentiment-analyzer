import re
import nltk
import spacy

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


# Load SpaCy model in english
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# List of stopwords
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Basic cleaning of text.
        - Remove HTML, punctuation, numbers, special characters
        - Lowercase the text
    """
   
    text = re.sub(r'<[^>]*>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text):
    """
    Tokenization using NLTK
    """
    return word_tokenize(text, language='english')

def remove_stopwords(tokens):
    """
    Remove stopwords from a list of tokens
    """
    return [word for word in tokens if word not in stop_words]

def lemmatize(tokens):
    """
    Lemmatization using SpaCy
    """
    doc = nlp(' '.join(tokens))
    return [token.lemma_ for token in doc if token.lemma_ != '-PRON-']

def preprocess_pipeline(text):
    """
    Complete pipeline for text preprocessing.
        - Clean the text
        - Tokenize the text
        - Remove stopwords
        - Lemmatize the tokens
    """
    text = clean_text(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    return ' '.join(tokens)
