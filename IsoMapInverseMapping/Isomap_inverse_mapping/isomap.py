import numpy as np
import pandas as pd
import sys

from sklearn.manifold import Isomap

    
def isomapping(dataset, neighbors_num, reduced_n):
    tags = ["chairs", "lamps", "tables"] 
    z = pd.read_csv('data' + '/' + 'z_vectors_' + tags[dataset-1] + '.csv', header = None).to_numpy()
    embedding = Isomap(n_neighbors = neighbors_num, n_components=reduced_n)
    z_transformed = embedding.fit_transform(z)
    #np.savetxt('data' + '/' + tags[dataset-1] + '_neighbors_' + str(neighbors_num) + '_isomap_reduced_' + str(reduced_n) +'.csv', z_transformed, delimiter = ',')
    return z_transformed
    

    

