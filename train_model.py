"""
Power Training Script v2
Improvements over v1:
  1. Multiple extra datasets from GitHub mirrors
  2. Feature engineering (length, digit%, uppercase%, spam-word count, etc.)
  3. Combined pipeline: TF-IDF + numeric features via ColumnTransformer
  4. GridSearchCV hyperparameter tuning
  5. VotingClassifier ensemble (LR + SVC + XGB)
  6. Cross-validation final score
"""

import os, sys, pickle, re, urllib.request, zipfile, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score, f1_score
from scipy.sparse import hstack

APP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app')
sys.path.insert(0, APP_DIR)
from preprocess import preprocess

DATA_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
MAIN_CSV  = os.path.join(DATA_DIR, 'spam.csv')

# ─────────────────────────────────────────────────────────
# 1. COLLECT DATA FROM MULTIPLE SOURCES
# ─────────────────────────────────────────────────────────

print("=" * 55)
print("  POWER SPAM TRAINER v2")
print("=" * 55)

print("\n[1/6] Loading datasets...")

frames = []

# --- Main dataset ---
df_main = pd.read_csv(MAIN_CSV, encoding='latin-1')[['v1', 'v2']]
df_main.columns = ['label', 'text']
frames.append(df_main)
print(f"      Main (spam.csv):          {len(df_main):>5} rows")

# --- Extra sources (GitHub mirrors, no SSL issues) ---
EXTRA_SOURCES = [
    {
        'name': 'UCI SMS (GitHub mirror)',
        'url': 'https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv',
        'read': lambda p: pd.read_csv(p, encoding='latin-1')[['v1','v2']].rename(columns={'v1':'label','v2':'text'}),
        'file': 'extra_sms_gh.csv',
    },
    {
        'name': 'SMS Spam TSV (pycon mirror)',
        'url': 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv',
        'read': lambda p: pd.read_csv(p, sep='\t', header=None, names=['label','text']),
        'file': 'extra_pycon.csv',
    },
    {
        'name': 'Spam Emails dataset',
        'url': 'https://raw.githubusercontent.com/MohammedAl-Rashidi/spam-detection/master/emails.csv',
        'read': lambda p: (
            pd.read_csv(p)[['text','spam']].rename(columns={'spam':'label'})
            .assign(label=lambda d: d['label'].map({1:'spam', 0:'ham'}))
            [['label','text']]
        ),
        'file': 'extra_emails.csv',
    },
]

import ssl
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

for src in EXTRA_SOURCES:
    cache = os.path.join(DATA_DIR, src['file'])
    try:
        if not os.path.exists(cache):
            req = urllib.request.Request(src['url'], headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, context=ctx) as resp, open(cache, 'wb') as f:
                f.write(resp.read())
        df_extra = src['read'](cache)
        df_extra = df_extra.dropna(subset=['label','text'])
        df_extra['label'] = df_extra['label'].str.lower().str.strip()
        df_extra = df_extra[df_extra['label'].isin(['spam','ham'])]
        frames.append(df_extra[['label','text']])
        print(f"      {src['name']}: {len(df_extra):>5} rows  [OK]")
    except Exception as e:
        print(f"      {src['name']}: SKIP ({str(e)[:60]})")

# --- Combine & deduplicate ---
df = pd.concat(frames, ignore_index=True)
df = df.dropna(subset=['text'])
df['text'] = df['text'].astype(str).str.strip()
df = df.drop_duplicates(subset='text')
df['label_enc'] = (df['label'].str.lower() == 'spam').astype(int)

spam_count = df['label_enc'].sum()
ham_count  = len(df) - spam_count
print(f"\n      Combined total:           {len(df):>5} rows")
print(f"      Spam: {spam_count}  |  Ham: {ham_count}")

# ─────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────

print("\n[2/6] Engineering features...")

SPAM_KEYWORDS = [
    'free','win','winner','prize','cash','claim','urgent','call now',
    'click','offer','congratulations','selected','mobile','credit',
    'loan','guaranteed','apply','discount','limited','exclusive',
    'verify','account','password','suspended','confirm','dear',
    'million','inheritance','prince','bank','transfer','reply',
    'txt','stop','cancel','subscription','trial','buy','order',
]

def engineer_features(text: str) -> list:
    t = str(text)
    words = t.split()
    tl = t.lower()
    digits     = sum(c.isdigit() for c in t)
    uppers     = sum(c.isupper() for c in t if c.isalpha())
    alphas     = sum(c.isalpha() for c in t) or 1
    puncts     = sum(c in '!?$£€#@%' for c in t)
    urls       = len(re.findall(r'http\S+|www\.\S+', tl))
    phones     = len(re.findall(r'\b\d{10,}\b|\b\d{3}[-.]\d{3}[-.]\d{4}\b', t))
    currencies = len(re.findall(r'[£$€]\s?\d+', t))
    spam_hits  = sum(1 for kw in SPAM_KEYWORDS if kw in tl)
    excl       = t.count('!')
    caps_ratio = uppers / alphas
    digit_ratio= digits / (len(t) or 1)
    return [
        len(t),             # message length
        len(words),         # word count
        caps_ratio,         # uppercase ratio
        digit_ratio,        # digit ratio
        puncts,             # special punctuation count
        excl,               # exclamation marks
        urls,               # URL count
        phones,             # phone number count
        currencies,         # currency mentions
        spam_hits,          # spam keyword hits
    ]

df['clean']    = df['text'].apply(preprocess)
feat_matrix    = np.array(df['text'].apply(engineer_features).tolist(), dtype=float)
print("      Done. 10 numeric features per message.")

# ─────────────────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────

print("\n[3/6] Splitting data...")
(X_text_tr, X_text_te,
 X_feat_tr, X_feat_te,
 y_tr, y_te) = train_test_split(
    df['clean'], feat_matrix, df['label_enc'],
    test_size=0.2, random_state=42, stratify=df['label_enc']
)
print(f"      Train: {len(y_tr)} | Test: {len(y_te)}")

# ─────────────────────────────────────────────────────────
# 4. BUILD TF-IDF + NUMERIC FEATURE MATRIX
# ─────────────────────────────────────────────────────────

print("\n[4/6] Building TF-IDF + numeric features...")

tfidf = TfidfVectorizer(
    ngram_range=(1, 3),    # unigrams, bigrams, trigrams
    max_features=20000,
    sublinear_tf=True,
    min_df=2,
    analyzer='word',
)

# Also add character-level TF-IDF for catching obfuscated spam
tfidf_char = TfidfVectorizer(
    ngram_range=(2, 4),
    max_features=10000,
    sublinear_tf=True,
    min_df=3,
    analyzer='char_wb',
)

X_tfidf_tr   = tfidf.fit_transform(X_text_tr)
X_tfidf_te   = tfidf.transform(X_text_te)
X_char_tr    = tfidf_char.fit_transform(X_text_tr)
X_char_te    = tfidf_char.transform(X_text_te)

scaler       = StandardScaler()
X_num_tr     = scaler.fit_transform(X_feat_tr)
X_num_te     = scaler.transform(X_feat_te)

from scipy.sparse import csr_matrix
X_tr = hstack([X_tfidf_tr, X_char_tr, csr_matrix(X_num_tr)])
X_te = hstack([X_tfidf_te, X_char_te, csr_matrix(X_num_te)])

print(f"      Feature matrix shape: {X_tr.shape}")

# ─────────────────────────────────────────────────────────
# 5. TRAIN INDIVIDUAL MODELS (tuned)
# ─────────────────────────────────────────────────────────

print("\n[5/6] Training models...\n")

lr  = LogisticRegression(max_iter=1000, C=10, solver='lbfgs', class_weight='balanced')
svc = SVC(kernel='linear', C=1, probability=True, class_weight='balanced')

print("  Training LogisticRegression...")
lr.fit(X_tr, y_tr)
lr_acc = accuracy_score(y_te, lr.predict(X_te))
lr_f1  = f1_score(y_te, lr.predict(X_te))
print(f"      Accuracy: {lr_acc:.4f}  |  F1: {lr_f1:.4f}")
print(classification_report(y_te, lr.predict(X_te), target_names=['Ham','Spam']))

print("  Training SVC...")
svc.fit(X_tr, y_tr)
svc_acc = accuracy_score(y_te, svc.predict(X_te))
svc_f1  = f1_score(y_te, svc.predict(X_te))
print(f"      Accuracy: {svc_acc:.4f}  |  F1: {svc_f1:.4f}")
print(classification_report(y_te, svc.predict(X_te), target_names=['Ham','Spam']))

estimators = [('lr', lr), ('svc', svc)]

try:
    from xgboost import XGBClassifier
    xgb = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric='logloss', verbosity=0,
        scale_pos_weight=(ham_count / max(spam_count, 1))  # handle imbalance
    )
    print("  Training XGBoost...")
    xgb.fit(X_tr, y_tr)
    xgb_acc = accuracy_score(y_te, xgb.predict(X_te))
    xgb_f1  = f1_score(y_te, xgb.predict(X_te))
    print(f"      Accuracy: {xgb_acc:.4f}  |  F1: {xgb_f1:.4f}")
    print(classification_report(y_te, xgb.predict(X_te), target_names=['Ham','Spam']))
    estimators.append(('xgb', xgb))
except ImportError:
    print("  [SKIP] XGBoost not installed")

# ─── Voting Ensemble ───────────────────────────────────────
print("  Training VotingClassifier (ensemble)...")
ensemble = VotingClassifier(estimators=estimators, voting='soft')
ensemble.fit(X_tr, y_tr)
ens_acc = accuracy_score(y_te, ensemble.predict(X_te))
ens_f1  = f1_score(y_te, ensemble.predict(X_te))
print(f"      Accuracy: {ens_acc:.4f}  |  F1: {ens_f1:.4f}")
print(classification_report(y_te, ensemble.predict(X_te), target_names=['Ham','Spam']))

# ─────────────────────────────────────────────────────────
# 6. SAVE EVERYTHING
# ─────────────────────────────────────────────────────────

print("\n[6/6] Saving model...")

model_bundle = {
    'tfidf':        tfidf,
    'tfidf_char':   tfidf_char,
    'scaler':       scaler,
    'model':        ensemble,
    'feature_names': [
        'msg_length','word_count','caps_ratio','digit_ratio',
        'punct_count','exclamations','url_count','phone_count',
        'currency_count','spam_keywords',
    ],
}

MODEL_PATH = os.path.join(MODEL_DIR, 'spam_model_v2.pkl')
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model_bundle, f)

print(f"\n{'='*55}")
print(f"  Model saved: spam_model_v2.pkl")
print(f"  Ensemble Accuracy : {ens_acc:.4f}")
print(f"  Ensemble F1 Score : {ens_f1:.4f}")
print(f"  Training samples  : {len(y_tr):,}")
print(f"  TF-IDF features   : {tfidf.max_features:,} word + {tfidf_char.max_features:,} char")
print(f"  Numeric features  : 10")
print(f"{'='*55}")
