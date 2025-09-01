#Program to test the performance of a model on an external test dataset - classification dataset

import sys
import csv
import pickle
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn import linear_model, metrics, ensemble
from scipy import stats
from scipy.stats import pearsonr
from tqdm import tqdm
from multiprocessing import Pool
import itertools
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error, roc_auc_score, average_precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, precision_score

test_data_pos = sys.argv[1]
test_data_neg = sys.argv[2]
dataset = sys.argv[3]
features_file_path = sys.argv[4]


df1 = pd.read_csv(test_data_pos, sep='\t', header=0)  #Test dataset positive
df1.fillna(0, inplace=True)

df2 = pd.read_csv(test_data_neg, sep='\t', header=0)  #Test dataset negative
df2.fillna(0, inplace=True)

with open(features_file_path,'r') as features_file:
    top_model_feat = [line.rstrip('\n') for line in features_file]

print("Top model features: "+','.join(top_model_feat))

model_dataset = pd.read_csv(dataset, sep='\t', header=0)
X_dataset = model_dataset[top_model_feat].to_numpy()
Y_dataset = model_dataset['pKd'].to_numpy()
model = linear_model.LinearRegression()
model.fit(X_dataset, Y_dataset)

pos_dataset = df1[top_model_feat].to_numpy()
pos_pred = model.predict(pos_dataset)
pos_true = [1] * pos_pred.shape[0]

neg_dataset = df2[top_model_feat].to_numpy()
neg_pred = model.predict(neg_dataset)
neg_true = [0] * neg_pred.shape[0]

pos_pred_classes = []
neg_pred_classes = []

pos_pred_binary = pos_pred>=4.0
neg_pred_binary = neg_pred>=4.0

for val in pos_pred:
	if(val>=4.0):
		pos_pred_classes.append(1)  #Correct prediction of actives - True positive
	else:
		pos_pred_classes.append(0)  #Incorrect prediction of actives - False negative

for val in neg_pred:
	if(val<4.0):
		neg_pred_classes.append(1)  #Correct prediction of inactives - True negative
	else:
		neg_pred_classes.append(0)  #Incorrect prediction of inactives - False positive

pred_all = []
true_all = []

pred_all.extend(pos_pred_binary)
pred_all.extend(neg_pred_binary)
#pred_all.extend(pos_pred_classes)
#pred_all.extend(neg_pred_classes)
true_all.extend(pos_true)
true_all.extend(neg_true)

tn, fp, fn, tp = confusion_matrix(true_all, pred_all).ravel()
print(tp, tn, fp, fn)

print("ROC-AUC score =", roc_auc_score(np.asarray(true_all), np.asarray(pred_all)))
print("Average precision =", average_precision_score(np.asarray(true_all), np.asarray(pred_all)))
print("Precision =", precision_score(np.asarray(true_all), np.asarray(pred_all)))
print("Specificity =", tn/(tn+fp))
print("Recall =", recall_score(np.asarray(true_all), np.asarray(pred_all)))
print("F1-score =", f1_score(np.asarray(true_all), np.asarray(pred_all)))
print("MCC =", matthews_corrcoef(np.asarray(true_all), np.asarray(pred_all)))