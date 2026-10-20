import os
import re
import pickle
import numpy as np

# ─── Load preprocessing ────────────────────────────────────────────────────────
from preprocess import preprocess

# ─── Load model bundle (v2) or fallback to v1 ─────────────────────────────────
_BASE = os.path.dirname(os.path.abspath(__file__))
_MODEL_V2 = os.path.join(_BASE, '..', 'model', 'spam_model_v2.pkl')
_MODEL_V1 = os.path.join(_BASE, '..', 'model', 'spam_model.pkl')

_tfidf = _tfidf_char = _scaler = _sklearn_model = None
_USE_V2 = False

if os.path.exists(_MODEL_V2):
    with open(_MODEL_V2, 'rb') as f:
        _bundle = pickle.load(f)
    _tfidf        = _bundle['tfidf']
    _tfidf_char   = _bundle['tfidf_char']
    _scaler       = _bundle['scaler']
    _sklearn_model= _bundle['model']
    _USE_V2 = True
    print("[INFO] Loaded model v2 (ensemble + feature engineering)")
else:
    with open(_MODEL_V1, 'rb') as f:
        _vectorizer, _sklearn_model = pickle.load(f)
    print("[INFO] Loaded model v1 (single classifier)")

# ─── Spam keyword list for numeric features ────────────────────────────────────
_SPAM_KEYWORDS = [
    'free','win','winner','prize','cash','claim','urgent','call now',
    'click','offer','congratulations','selected','mobile','credit',
    'loan','guaranteed','apply','discount','limited','exclusive',
    'verify','account','password','suspended','confirm','dear',
    'million','inheritance','prince','bank','transfer','reply',
    'txt','stop','cancel','subscription','trial','buy','order',
]

def _engineer_features(text: str) -> list:
    t  = str(text)
    tl = t.lower()
    words   = t.split()
    digits  = sum(c.isdigit() for c in t)
    uppers  = sum(c.isupper() for c in t if c.isalpha())
    alphas  = sum(c.isalpha() for c in t) or 1
    puncts  = sum(c in '!?$£€#@%' for c in t)
    urls    = len(re.findall(r'http\S+|www\.\S+', tl))
    phones  = len(re.findall(r'\b\d{10,}\b|\b\d{3}[-.]\d{3}[-.]\d{4}\b', t))
    currs   = len(re.findall(r'[£$€]\s?\d+', t))
    hits    = sum(1 for kw in _SPAM_KEYWORDS if kw in tl)
    excl    = t.count('!')
    return [
        len(t), len(words), uppers / alphas, digits / (len(t) or 1),
        puncts, excl, urls, phones, currs, hits,
    ]

# ─── Load Transformer model (BERT-tiny fine-tuned on SMS spam) ────────────────
_transformer = None
try:
    from transformers import pipeline as hf_pipeline
    print("[INFO] Loading BERT transformer model...")
    _transformer = hf_pipeline(
        "text-classification",
        model="mrm8488/bert-tiny-finetuned-sms-spam-detection",
        truncation=True,
        max_length=128
    )
    print("[INFO] Transformer model loaded successfully.")
except Exception as e:
    print(f"[WARN] Transformer not available, using sklearn only: {e}")


# ─── Public predict function ───────────────────────────────────────────────────
def predict_spam(text: str) -> dict:
    """
    Returns:
      label      : 'Spam' | 'Not Spam'
      is_spam    : bool
      confidence : float (0–100)
      method     : 'transformer' | 'ensemble-v2' | 'sklearn'
      risk_level : 'High' | 'Medium' | 'Low'
    """
    if _transformer:
        try:
            result    = _transformer(text[:512])[0]
            is_spam   = result['label'].upper() == 'SPAM'
            raw_conf  = result['score']
            spam_conf = raw_conf if is_spam else (1 - raw_conf)

            # Cross-check with sklearn for borderline cases
            sk_spam, sk_conf, _ = _sklearn_predict(text)
            if abs(spam_conf - 0.5) < 0.15:   # borderline — blend scores
                spam_conf = 0.6 * spam_conf + 0.4 * sk_conf
                is_spam   = spam_conf >= 0.5

            method = 'transformer'
        except Exception:
            is_spam, spam_conf, method = _sklearn_predict(text)
    else:
        is_spam, spam_conf, method = _sklearn_predict(text)

    confidence = round(spam_conf * 100, 1)
    label      = 'Spam' if is_spam else 'Not Spam'
    risk_level = (
        'High'   if spam_conf >= 0.75 else
        'Medium' if spam_conf >= 0.40 else
        'Low'
    )

    return {
        'label':      label,
        'is_spam':    is_spam,
        'confidence': confidence,
        'method':     method,
        'risk_level': risk_level,
    }


def _sklearn_predict(text: str):
    """sklearn pipeline predict with v2 or v1 bundle."""
    clean = preprocess(text)

    if _USE_V2:
        from scipy.sparse import hstack, csr_matrix
        X_tfidf = _tfidf.transform([clean])
        X_char  = _tfidf_char.transform([clean])
        X_num   = _scaler.transform([_engineer_features(text)])
        X       = hstack([X_tfidf, X_char, csr_matrix(X_num)])
        method  = 'ensemble-v2'
    else:
        X      = _vectorizer.transform([clean])
        method = 'sklearn'

    is_spam   = bool(_sklearn_model.predict(X)[0])
    proba     = _sklearn_model.predict_proba(X)[0]
    spam_conf = float(proba[1])
    return is_spam, spam_conf, method
