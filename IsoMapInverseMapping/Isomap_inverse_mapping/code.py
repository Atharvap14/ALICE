import sys
import numpy as np
import time
from isomap import *
from rbf_inverse_mapping import *
import matplotlib.pyplot as plt

#dataset = int(sys.argv[1])
dataset = 1
#neighbors_num = int(sys.argv[2])
neighbors_num = 7
#reduced_n = int(sys.argv[3])
reduced_n = 2

tags = ["chairs", "lamps", "tables"] 

st_time = time.time()
iso_z_2 = isomapping(dataset, neighbors_num, reduced_n)
ed_time = time.time()

## Code Modifications by Atharva to Visualize ISOMAP
# print(iso_z_2.shape)

# plt.scatter(iso_z_2[:,0],iso_z_2[:,1])
# plt.show()
print("Time to isomap " + tags[dataset-1] + " dataset : {}".format(ed_time - st_time))

new_z_2 = pd.read_csv('data' + '/' + 'chairs_neighbors_' + str(neighbors_num) + '_isomap_reduced_' + str(reduced_n) + '_test_points_111_360.csv', header = None).to_numpy()

st_time = time.time()
sample_chairs = inverse(dataset, neighbors_num, reduced_n, iso_z_2, new_z_2)

ed_time = time.time()
print("Time to inverse mapping " + tags[dataset-1] + " dataset : {}".format(ed_time - st_time))
