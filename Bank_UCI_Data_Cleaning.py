# Import Packages
import time
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

#Just to suppress some warnings about copying data on a slice of a dataframe
import warnings
warnings.filterwarnings("ignore")

line = "------------------------------------------------------------------\n"

# Import Data
print ("\n" , line)
print ("\nReading csv...\n")
df = pd.read_csv('./bank-additional/bank-additional-full.csv',delimiter=';')
df = df.drop('duration',axis=1)
print("Size of dataset - ",len(df))
print("\nColumns - ", list(df.columns))

#encoding y
df['y'] = df['y'].map({'yes':1,'no':0})

# Create categorical variables from numerical

# Convert pdays into categorical
print (line)
print ("Creating Categorical Variables from Numerical...")
df.loc[df["pdays"] == 999, "pdayscat"] = 0 #cutomer was not contacted for previous campaign
df.loc[df["pdays"] != 999, "pdayscat"] = 1 #cutomer was contacted for previous campaign

# Convert age into categorical
l1 = 23
l2 = 62
df.loc[df["age"] <= l1, "agecat"] = 'young'
df.loc[((df["age"] > l1) & (df["age"] < l2)), "agecat"] = 'adult'
df.loc[df["age"] >= l2, "agecat"] = 'senior'

# Convert Campaign into categorical
l1 = 12
df.loc[df["campaign"] <= l1, "campcat"] = 'low'
df.loc[df["campaign"] > l1, "campcat"] = 'high'

# Types of attributes
numeric = ['age', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 
           'euribor3m', 'nr.employed']
binary = ['default', 'housing', 'loan','contact','campcat']
categorical = ['poutcome', 'job', 'marital', 'month','day_of_week','agecat']
hierarchy = ['education']

print ("\nSearching for null values in cateogirical variables...\n")
print ("Null Values - \n")
#Null Columns
null_cols = []
for column in binary+categorical+hierarchy:
    missing_count = len(df.loc[df[column] == 'unknown']['y'])
    
    if missing_count < 500:
        print (column, " - ", missing_count)
        df = df[df[column] != 'unknown'] #Delete the rows
    elif missing_count > 500:
        #Find which columns have still retained unkowns
        number = missing_count
        print("{} - {}".format(column,number))
        null_cols.append(column) #Populate the list
        
print(line)

print ("Encoding Categorical Variables...\n")

#Hierarchical Encoding for education
""" This is specific for education. We need a manually created hierarcy as an input here."""
values = ["illiterate", "basic.4y", "basic.6y", "basic.9y", "high.school",  "professional.course", "university.degree"]
levels = range(1,len(values)+1)
dictionary = dict(zip(values,levels))
df['education']=df['education'].map(dictionary)
df['education'] = df['education'].replace('unknown', np.nan, regex=True)


#Categorical Encoding
def categorical_encoding(df,categorical=categorical):
    for column in categorical:
        
        dummies = pd.get_dummies(df[column])
        dummies = dummies.rename(columns = lambda x: column + '_' + str(x))
        
        df = pd.concat([df,dummies],axis=1)
        df = df.drop(column,axis=1)
    return df

df = categorical_encoding(df)

#Binary Encoding
for column in binary:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    
    #Convert 'unknown' into np.nan
    l = list(le.classes_)
    try:
        i = l.index('unknown')
        df.loc[df[column] == i, column] = np.nan
    except:
        pass
    
print (line)

print ("Null Values in numerical columns - \n")
for column in numeric:
    null = sum(df[column].isnull())
    print (" {} - {}".format(column, str(null)))
    if null > 0:
        null_cols.append(column)
        

#Using random forest to fill null values
t = time.time()

def predict_unknown(trainX, trainY, testX):
    """ Predicting unknown data using random forest"""
    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit(trainX, trainY)
    test_predictY = forest.predict(testX).astype(int)
    return pd.DataFrame(test_predictY,index=testX.index)

for column in null_cols:
    
    test_data = df[df[column].isnull()]
    testX = test_data.drop(null_cols, axis=1)
    train_data = df[df[column].notnull()]        
    trainY = train_data[column]
    trainX = train_data.drop(null_cols, axis=1)
    #print(trainX.isnull().sum())
    #print(trainY.value_counts())
    test_data[column] = predict_unknown(trainX, trainY, testX)
    df = pd.concat([train_data, test_data])
    print(column, end=' ')
    
print ("Time takes for filling null value is {} seconds".format(time.time() - t))
print (line)

#Write the cleaned data to a csv
df.to_csv("clean_data_52_features.csv",index=False)        
