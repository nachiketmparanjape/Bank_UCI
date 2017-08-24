# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 13:46:41 2017

@author: Nachiket
"""
# Import Packages
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import recall_score, f1_score, roc_curve, auc
from sklearn.svm import SVC, LinearSVC

Xtest = []
ytest = []

#Functions
def normalizer(traindata,testdata):
    """ This is used to normalize the data between 0 to 1 """    
    
    trainnormalized = (traindata-min(traindata))/(max(traindata)-min(traindata))
    testnormalized = (testdata-min(traindata))/(max(traindata)-min(traindata))
    return trainnormalized, testnormalized

def cv_score(clf, x, y, score_func=f1_score):
    """ Apply 5-fold cross validation on the training data
        Default Scoring : F1-score """
    result = 0
    nfold = 5
    for train, test in KFold(nfold,random_state=0).split(x): # split data into train/test groups, 5 times
        clf.fit(x[train], y[train]) # fit
        result += score_func(y[test],clf.predict(x[test])) # evaluate score function on held-out data
    return result / nfold # average
    
def roccurve(trainedevaluator,Xtest=Xtest,ytest=ytest):
    """ Plots ROC Curve and returns area under the curve """
    
    #Find probabilities
    preds = trainedevaluator.predict_proba(Xtest)[:,1]
    
    #ROC
    fpr, tpr, _ = roc_curve(ytest, preds)
    
    #Area Under the curve
    area = auc(fpr,tpr)
    
    #Plot
    fig = plt.figure(figsize=(10,6))
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("Ttrue Positive Rate")
    plt.plot(fpr,tpr)
    plt.plot([0,1],[0,1],'--',alpha=0.5)
    fig.savefig("ROC_Curve.jpeg")
    
    return area


def main():
    
    #Design
    line = "\n---------------------------------------------------------------------------\n"
    
    #Import Data
    print ("\n\nImporting Data..\n")
    t = time.time()
    df = pd.read_csv("clean_data_basic_pdayscat_agecat.csv",index_col=False)
    print("%f seconds" % (time.time() - t))
    
    print ( "{}% positives in the data".format(round(len(df.loc[df.y == 1])*100.0/len(df),2)))
    
    
    #Features
    features = df.drop('y',axis=1)

    #Split the data in traing and test data
    X, Xtest, y, ytest = train_test_split(features, df['y'],random_state=5,test_size=0.3)
    
    print ( "\n{}% positives in the test data".format(round(len(ytest.loc[ytest == 1])*100.0/len(ytest),2)))
    
    print (line)
    t = time.time()
    print ("Normalizing the data..")
    print("%f seconds" % (time.time() - t))
    print (line)
    
    print ("Tuning SVM using Grid Search..\n")
    
    parameters = {'C':[10,100],  'gamma': 
              [0.01,1,10], 'degree':[1,2,3]}
    # 
    svr = SVC(kernel='rbf')
    t = time.time()
    grid = GridSearchCV(svr, parameters, scoring='recall')
    print("grid : %f seconds" % (time.time() - t))
    t = time.time()
    grid.fit(X.values, y.values)
    print("fit : %f seconds" % (time.time() - t))
    
    print("\n")
    bestimator = grid.best_estimator_
    print(bestimator)
    print ("\n")
    print(grid.best_params_)
    print(line)
    
    print("Training using the best estimator")
    svcf = grid.bestimator
    svcf.fit(X,y)
    
    print("\nMetrics")
    preds = svcf.predict(Xtest)
    
    accuracy = accuracy_score(ytest,preds)
    recall = recall_score(ytest,preds)
    f1 = f1_score(ytest,preds)
    area = roccurve(svcf)
    
    print ("\n1.Accuracy = {}".format(accuracy))
    print ("\n2.Recall = {}".format(recall))
    print ("\n3.F1 Score = {}".format(f1))
    print ("\n4.ROC AUC = {}".format(area))
 

#Run SVC Run
main()