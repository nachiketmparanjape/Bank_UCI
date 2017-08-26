#Import packages
import time
import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, accuracy_score, recall_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
#Import Cleaned Data
line = "------------------------------------------------------------------\n"
print ("\n",line)
print ("Importing cleaned data..\n")
df = pd.read_csv("clean_data_52_features.csv",index_col=False)

#Import Centroid column
centroids = pd.read_csv("centroids.csv")
df = pd.concat([df, centroids], axis=1)

features = df.drop('y',axis=1) #features

print ( "There are {}% positives in the data.\n".format(round(len(df.loc[df.y == 1])*100.0/len(df),2)))
print (line)

print ("Splitting data into training and test sets...\n")
X, Xtest, y, ytest = train_test_split(features, df['y'],random_state=1,test_size=0.3)

#Define some funcitons
    
def roccurve(trainedevaluator,plotting=True,X=X,y=y,Xtest=Xtest,ytest=ytest):
    """ Plots ROC Curve and return area under the curve """
    
    #fit
    trainedevaluator.fit(X,y)
    
    #Find probabilities
    preds = trainedevaluator.predict_proba(Xtest)[:,1]
    
    #ROC
    fpr, tpr, _ = roc_curve(ytest, preds)
    
    #Area Under the curve
    area = auc(fpr,tpr)
    
    return area

def prediction(clf,X=X,y=y,Xtest=Xtest,ytest=ytest):
    """ Fits model and prints out a specific range of evaluation criteria """
    clf.fit(X.values, y.values)
    preds = clf.predict(Xtest)
    cm = confusion_matrix(ytest, preds)
    
    print ("\nClassification Report on the Training data -\n", classification_report(y, clf.predict(X)))
    
    print("Accuracy Score =", accuracy_score(ytest, preds))
    print("\nConfusion Matrix -\n",cm)
    print("\nClassification Report -\n",classification_report(ytest, preds))
    try:
        print("Area under the ROC curve = {}".format(roccurve(clf)))
    except AttributeError:
        print ("Cannot draw ROC - predict_proba is not an attribute")

# Random Forest

rf = RandomForestClassifier()

parameters = {'n_estimators':[20,30,50,100,200],
              'max_features':['sqrt','log2',None],
             'criterion':['gini','entropy'],
             'min_samples_split':[2,10,0.01]}

t = time.time()
gs = GridSearchCV(rf,parameters,cv=5,scoring='recall',n_jobs=-1) #GridSearch
t = time.time()
print("\n %f seconds" % (time.time() - t))

t = time.time()
prediction(gs) #Fit, Predict and Evaluate
print("\n %f seconds" % (time.time() - t))