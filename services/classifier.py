import joblib
def load_models():
    return {
        'tfidf': joblib.load('tfidf.pkl'),
        'clf_intent': joblib.load('clf_intent.pkl'),
        'clf_sev': joblib.load('clf_sev.pkl'),
        'le_intent': joblib.load('le_intent.pkl'),
        'le_sev': joblib.load('le_sev.pkl')
    }
