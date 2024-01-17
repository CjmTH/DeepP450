import numpy as np
from sklearn import metrics
import pandas as pd

soft_voting = pd.read_csv('total_soft.csv')

def soft_voting_metric():
    y_true = soft_voting['true_label']
    y_pred = soft_voting['final_label']
    y_pred_pro = soft_voting['mean_prob']

    ACC = metrics.accuracy_score(y_true, y_pred)
    BACC = metrics.balanced_accuracy_score(y_true, y_pred)
    F1 = metrics.f1_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred)
    auc = metrics.roc_auc_score(y_true, y_pred_pro)
    ap = metrics.average_precision_score(y_true, y_pred_pro)
    jaccard = metrics.jaccard_score(y_true, y_pred)

    print("ACC: ", ACC)
    print("BACC: ", BACC)
    print("F1: ", F1)
    print("recall: ", recall)
    print("precision: ", precision)
    print("auc: ", auc)
    print("ap: ", ap)
    print("jaccard: ", jaccard)



if __name__ == '__main__':
    print("soft_voting_metric:")
    soft_voting_metric()
    print("")




