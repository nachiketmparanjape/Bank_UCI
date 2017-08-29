""" Clustering for creating features """

#Import packages
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.decomposition import PCA


df = pd.read_csv('clean_data_52_features.csv') #Import Data

y = df.y
df = df.drop('y',axis=1)

colormap = np.array(sns.color_palette('Set1',20))
#colormap = np.array(['k','r','g','y','b'])

#Normalize the data
def normalizer(data):
    
    normalized = (data-min(data))/(max(data)-min(data))
    return normalized

normdf = pd.DataFrame()

for column in df:
    normdf[column] = normalizer(df[column])
    

# K-means Clustering

krange = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

data = normdf.values
# Dimensionality reduction
pca = PCA(n_components=2)
data2 = pca.fit_transform(data)


sillhoutte_scores = []

for n_cluster in krange:

    km = KMeans(n_clusters=n_cluster, algorithm='full')
    
    l = km.fit_predict(data)
    

    
    ss = metrics.silhouette_score(data,l)
    sillhoutte_scores.append(ss)
    
    #Plot
    preds = km.labels_
    centroids = km.cluster_centers_
    
    
    
    
    # Plot Orginal
    fig = plt.figure(figsize=(16,6))
    
    
    plt.subplot(1, 2, 1)
    
    
#    plt.xticks(np.arange(0,1.1,0.1))
#    plt.yticks(np.arange(0,1.1,0.1))
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    
    x1 = list(data2[:,0])
    y1 = list(data2[:,1])
    
    plt.xlim((min(x1),max(x1)))
    plt.ylim((min(y1),max(y1)))
    
    plt.scatter(x=x1,y=y1, s=10, c=colormap[y], alpha = 0.2)
    p1 = plt.scatter([50,100], [50,100], marker='o', color=colormap[0])
    p2 = plt.scatter([50,100], [50,100], marker='o',color=colormap[1])
    p3 = plt.scatter([50,100], [50,100], marker='*',color='blue')
    plt.legend((p1,p2),('No','Yes'),numpoints=1, loc=1, fontsize=15)
    plt.title('Scatter Plot')
     
    # Plot Predicted with corrected values
    plt.subplot(1, 2, 2)
    
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    
    plt.scatter(x=x1, y=y1, c=colormap[preds], s=10, alpha=0.2)
#    plt.scatter(x=centroids[:,0],y=centroids[:,1], s=100, c='blue', marker='*')
    plt.title('Clusters and Centroids')
    
    plt.savefig("./Images/Scatter_" + str(n_cluster) + ".png")
    


print (sillhoutte_scores)

plt.plot(x=sillhoutte_scores,y=krange)
    
    
    