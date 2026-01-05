from sklearn.metrics import f1_score, classification_report, confusion_matrix

def compute_metrics(y_true, y_pred):
    return {"macro_f1": f1_score(y_true, y_pred, average="macro")}

def classification_report_text(y_true, y_pred):
    return classification_report(y_true, y_pred, digits=4)

def confusion_matrix_array(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)