def fuse_confidence(intent_conf, sev_conf, kb_hits_count):
    kb_score = min(kb_hits_count/3, 1.0)
    return round((intent_conf*0.6 + sev_conf*0.2 + kb_score*0.2), 2)
