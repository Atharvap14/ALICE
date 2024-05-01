# from utils import *
# import numpy as np
# import argparse
# from Models.IMNET.modelAE import *
# import torch
# ## Let us code the main first without the GUI layer. 
# # TO DO : Add GUI layer

# # Reading Training Latents
# filename = r"Models\IMNET\checkpoint\all_vox256_img_ae_64\all_vox256_img_train_z.hdf5"
# z_latents=read_hdf5(filename=filename)[:400]

# ## Hyperparameters
# neighbors_num = 7
# reduced_n = 2

# iso_z_2 = isomapping(z_latents, neighbors_num, reduced_n)
# min_z=np.min(iso_z_2,axis=0)
# max_z=np.max(iso_z_2,axis=0)
# print("Min Z",min_z)
# print("Max Z",max_z)

# new_x=np.random.uniform(min_z[0],max_z[0])
# new_y=np.random.uniform(min_z[1],max_z[1])

# print("New X",new_x)
# print("New Y",new_y)

# new_z_2 = np.array([[new_x,new_y]])

# sample_chairs = inverse(z_latents, iso_z_2, new_z_2)
# sample_chairs = sample_chairs.astype(float)


# args = argparse.Namespace()

# # Assign default values to the attributes
# args.epoch = 0
# args.iteration = 0
# args.learning_rate = 0.00005
# args.beta1 = 0.5
# args.dataset = "all_vox256_img"
# args.checkpoint_dir = "./IMNET/checkpoint"
# args.data_dir = "./IMNET/data/all_vox256_img/"
# args.sample_dir = "./IMNET/samples/"
# args.sample_vox_size = 64
# args.train = False
# args.start = 0
# args.end = 16
# args.ae = True
# args.svr = False
# args.getz = False


# modelAE = IM_AE(args)
# modelAE.test_z(args,sample_chairs,256)

# ## This End to End code for prototyping the method. It works well and now I just have to setup a GUI for input and output.

from BaseComponents.GUILayer import *

view = GUI()
