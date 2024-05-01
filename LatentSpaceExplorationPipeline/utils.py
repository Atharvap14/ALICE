import h5py
import numpy as np
import pandas as pd
import sys
from sklearn.manifold import Isomap
import numpy as np
import pandas as pd
import math

def euclidean(x, y):
    dist = np.sqrt(np.sum(np.square(np.subtract(x,y)),keepdims=True))
    return dist   

def inverse(X, iso_z, new_z):
    
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
    

    return z_recons

def inverse_map(X, iso_z):
    Y = iso_z

    [n, D] = X.shape
    [N2, d2] = Y.shape
    K = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            K[i,j] = euclidean(Y[i,:], Y[j,:])

    A = np.linalg.solve(K,X)
    return (A,n,D,Y)

def invert(new_z,coeffs):
    A,n,D,Y=coeffs
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
    

    return z_recons



def isomapping(z, neighbors_num, reduced_n):
    
    embedding = Isomap(n_neighbors = neighbors_num, n_components=reduced_n)
    z_transformed = embedding.fit_transform(z)
    return z_transformed,embedding
    

    

## Reads the first key value!!! TO:DO see how to make it generalizable
def read_hdf5(filename):
    with h5py.File(filename, "r") as f:
        a_group_key = list(f.keys())[0]
        ds_arr = f[a_group_key][()]  # returns as a numpy array
    return ds_arr

def write_hdf5(filename,key,data):
    """
    filename: Path to the file. ex: "../content/sample_data/h5inputs.hdf5"
    key: The key for dictionary
    data: A numpy array that is the value corresponding to key
    """
    fs = h5py.File(filename, 'w')
    dset = fs.create_dataset(key, data=data)
    fs.close()