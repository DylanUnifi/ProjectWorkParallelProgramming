from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


def log_metrics(y_true, y_pred):
    """Compute classification metrics.

    ``precision_score`` and ``recall_score`` use ``zero_division=0`` to suppress
    warnings when a class has no predicted or true samples.
    """

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    return acc, f1, precision, recall
