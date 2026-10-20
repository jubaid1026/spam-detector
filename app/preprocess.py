import re
import string
import nltk

# Download required NLTK data (runs once)
for resource in ['stopwords', 'punkt', 'wordnet', 'omw-1.4']:
    try:
        nltk.data.find(f'corpora/{resource}' if resource not in ['punkt'] else f'tokenizers/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

_stop_words = set(stopwords.words('english'))
_lemmatizer = WordNetLemmatizer()

def preprocess(text: str) -> str:
    """Full preprocessing pipeline: clean -> tokenize -> remove stopwords -> lemmatize."""
    if not isinstance(text, str):
        return ''

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', ' url ', text)

    # Replace phone numbers with token
    text = re.sub(r'\b\d{10,}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b', ' phonenumber ', text)

    # Replace currency amounts
    text = re.sub(r'[£$€]\s?\d+[\d,]*', ' moneyamount ', text)

    # Remove punctuation and digits
    text = re.sub(r'[^a-z\s]', ' ', text)

    # Tokenize
    tokens = text.split()

    # Remove stopwords and short tokens, then lemmatize
    tokens = [
        _lemmatizer.lemmatize(t)
        for t in tokens
        if t not in _stop_words and len(t) > 1
    ]

    return ' '.join(tokens)
