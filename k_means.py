import numpy as np
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import collections
from scipy.spatial.distance import cdist

# normalizasyon fonksiyonu                    
def normalization(data):     
    min = np.min(data)
    max = np.max(data)
    for datum in range(0,len(data)):
        new = (data[datum]-min)/(max-min)
        data[datum] = new
    return data
 
K= int(input("Kume Sayisi: "))


data = pd.read_csv(r"Final-data.csv")

# veriler normalize edilmesi
for key in data:
    data[key] = normalization(data[key])   

df = pd.DataFrame(data, columns=['a1', 'a2', 'a3' , 'a4' , 'a6', 'a7', 'a8','a9'])


# küme sayısına göre kümelemeyi yapar ve WCSS değerini hesaplar
kmeans = KMeans(init="random", n_clusters=K, n_init=10, random_state=1)
kmeans.fit(df)
WCSS = kmeans.inertia_ 
clusters = kmeans.labels_ 

# kümelerdeki veri sayısını hesaplar
counter = collections.Counter(clusters)
centroids = kmeans.cluster_centers_

#  TSS HESAPLAMA
TSS = 0
avarage = np.sum(np.sum(df)) / np.size(df)

for column in df:
    for data in df[column]:
        TSS += pow(data-avarage,2)

# BCSS HESAPLAMA
BCSS = TSS - WCSS


        # dunn index
# matrisler arası mesafeyi hesaplar
distance_matrix = cdist(df, df, metric='euclidean')

# Kümeler içi min. mesafeyi hesaplar
min_distance = np.inf
for i in range(K):
    for j in range(K):
        if i != j:
            distance = np.inf
            for i in df[clusters==i].index:
                for j in df[clusters==j].index:
                    d = distance_matrix[i, j]
                    if d < distance:
                      distance = d
            if distance < min_distance:
                min_distance = distance
 
# kümeler arası mak. mesafeyi hesaplar
max_distance = 0
for cluster in set(clusters):
    cluster_points = df[clusters==cluster]
    diameter = np.max(cdist(cluster_points, cluster_points, metric='euclidean'))
    if diameter > max_distance:
        max_distance = diameter


dunn_index = min_distance / max_distance


# sonuc.txt ye yazma 
f = open("sonuc.txt", "w")
for i in range(len(clusters)):
    f.write(f"Kayit {i+1}:   Kume {clusters[i]+1}\n")
for i in counter:    
    f.write(f"\nKume {i+1}: {counter[i]} Kayit")
f.write(f"\n\nWCSS: {WCSS}\nBCSS: {BCSS}\nDunn Index: {dunn_index}")    
   
f.close()

quest = str(input("Veri Görselleştirme İstiyor musunuz ? (y/n)"))

# verileri görselleştirme
if(quest=="y"):
    x_label = str(input("X Ekseni: "))
    y_label = str(input("y Ekseni: "))

    plt.scatter(df[x_label], df[y_label] ,c= kmeans.labels_.astype(float), s=50, alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='r',marker='*', s=50)  
    plt.show()
    
