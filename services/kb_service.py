import joblib, numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_kb(tfidf_path='tfidf.pkl', kb_df_path='kb_df.pkl', kb_vec_path='kb_vec.pkl'):
    tfidf = joblib.load(tfidf_path)
    kb_df = joblib.load(kb_df_path)
    kb_vec = joblib.load(kb_vec_path)
    return tfidf, kb_df, kb_vec

def query_kb(tfidf, kb_df, kb_vec, text, top_n=3, threshold=0.2):
    v = tfidf.transform([text])
    sims = cosine_similarity(v, kb_vec)[0]
    idxs = list(np.argsort(-sims)[:top_n])
    hits = []
    for i in idxs:
        if sims[i] > threshold:
            row = kb_df.iloc[i].to_dict()
            row['score'] = float(sims[i])
            hits.append(row)
    return hits
