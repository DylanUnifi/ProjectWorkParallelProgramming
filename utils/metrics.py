from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

def log_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return acc, f1, precision, recall
