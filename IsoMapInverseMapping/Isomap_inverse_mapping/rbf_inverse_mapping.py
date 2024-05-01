import numpy as np
import pandas as pd
import math

def euclidean(x, y):
    dist = np.sqrt(np.sum(np.square(np.subtract(x,y)),keepdims=True))
    return dist   

def inverse(dataset, neighbors_num, reduced_n, iso_z, new_z):
    tags = ["chairs", "lamps", "tables"]
    X = pd.read_csv('data' + '/' + 'z_vectors_' + tags[dataset-1] + '.csv', header = None).to_numpy()
    Y = iso_z

    [n, D] = X.shape
    [N2, d2] = Y.shape
    K = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            K[i,j] = euclidean(Y[i,:], Y[j,:])

    A = np.linalg.solve(K,X)
    Y_new = new_z
    [R,C] = Y_new.shape

    for i in range(n-R):
        Y_new = np.concatenate((Y_new, [Y_new[R-1,:]]), axis=0)

    X_new = np.zeros([n,D])

    K_new = np.zeros([n,n])

    for i in range(n):
        for j in range(n):
            K_new[i,j] = euclidean(Y_new[i,:], Y[j,:])

    for i in range(n):
        for j in range(D):
            X_new[i,j] = np.dot(np.transpose(A[:,j]),np.transpose(K_new[i,:]))
         

    z_recons = np.zeros([R,D])
    for i in range(R):
        for j in range(D):
            z_recons[i,j] = X_new[i,j]
    
    name = 'outputs' + '/' + tags[dataset-1] + '_neighbors_' + str(neighbors_num) + '_isomap_reduced_' + str(reduced_n) + '_inversed' + '.csv'
    np.savetxt(name, z_recons, delimiter = ',')
    return z_recons
