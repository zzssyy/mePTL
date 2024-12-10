# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 07:57:43 2023

@author: Z
"""
from sklearn.cluster import KMeans
import numpy as np
import random

#Divide subset through k-means, features as input
def getSubset(X, y, k=2, p=2):
    data_new = {}
    data_dict = {}
    features_neg = []
    for key, value in X.items():
        value_new = np.column_stack((value, y))
        X[key] = value_new
        if len(features_neg) == 0:
            features_neg = np.numpy(features_neg + value[np.where((y == 1))].tolist())
        else:
            features_neg = np.column_stack((features_neg, value))
    kmeans = KMeans(n_clusters=k, random_state=0).fit(features_neg)
    for i in range(0, p):
        index = np.where((kmeans.labels_==i))
        for key,value in X.items():
             pos = value[np.where((y==1))]
             neg = value[np.where((y==0))]
             neg_patition_index = np.random.choice(neg[index], len(index)/2)
             neg_patition = neg[neg_patition_index]
             data_new[key] = np.concatenate((pos, neg_patition))
        data_dict[str(i)] = data_new        
    return data_dict

#Divide subset directly
def getSubsetDir(X, y):
    X, y = np.array(X), np.array(y)    
    data_dict = {}
    data = np.column_stack((X, y))
    pos = data[np.where((y==1))]
    neg = data[np.where((y==0))]
    k = int(len(neg)/len(pos))
    
    for j in range(0, k):
        neg_patition_index = np.random.choice(neg.shape[0], len(pos))
        neg_patition = neg[neg_patition_index,:]
        neg_reminder_index = np.delete(np.arange(len(neg)), neg_patition_index)
        neg_reminder = neg[neg_reminder_index]
        data_dict[str(j)] = np.concatenate((pos, neg_patition))
    return data_dict

def SubsetSelection(X, acc):
    print("acc = ", acc)
    acc_mean = []
    acc_std = []
    index = 0
    for r in acc:
        acc_mean.append(np.mean(r))
        acc_std.append(np.std(r))
    acc_mean = np.array(acc_mean)
    acc_std = np.array(acc_std)
    acc_mean_ranked_index = [i[0] for i in sorted(enumerate(acc_mean), key=lambda x:x[1], reverse = True)]
    acc_mean_ranked_value = [i[1] for i in sorted(enumerate(acc_mean), key=lambda x:x[1], reverse = True)]
    if acc_mean_ranked_value[0] - acc_mean_ranked_value[1] >= 1:
        index = acc_mean_ranked_index[0]
    else:
        acc_std_ranked = [i[0] for i in sorted(enumerate(acc_std), key=lambda x:x[1])]
        index = acc_std_ranked[0]
    print("index=", index)
    train_features, train_labels, train_features_name = X[str(index)]
    return train_features, train_labels, train_features_name, index