#Program to test the performance of a model on an external test dataset - regression dataset

import sys
import csv
import pickle
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn import linear_model, metrics, ensemble
from scipy import stats
from scipy.stats import pearsonr
import itertools
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error

data = sys.argv[1]
dataset = sys.argv[2]
features_file_path = sys.argv[3]
outfile = sys.argv[4]

with open(features_file_path,'r') as features_file:
    top_model_feat = [line.rstrip('\n') for line in features_file]
    
print("Top model features: "+','.join(top_model_feat))

final_df = pd.read_csv(data, sep="\t", header=0)

model_dataset = pd.read_csv(dataset, sep='\t', header=0)
X_dataset = model_dataset[top_model_feat].to_numpy()
Y_dataset = model_dataset['pKd'].to_numpy()
model = linear_model.LinearRegression()
model.fit(X_dataset, Y_dataset)

Y_test = final_df['pKd'].to_numpy()
test_X = final_df[top_model_feat].to_numpy()
Y_pred = model.predict(test_X)
print(pearsonr(Y_test, Y_pred), mean_absolute_error(Y_test, Y_pred))

print("Y_test:", Y_test)
print("Y_pred:", Y_pred)

out = open(outfile, "w")
for val1, val2 in zip(Y_test, Y_pred):
	print(val1, val2, file=out)






















