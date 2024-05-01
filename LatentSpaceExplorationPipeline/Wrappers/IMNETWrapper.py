from BaseComponents.BaseEncoder import *
from BaseComponents.BaseDecoder import *
import argparse
from Models.IMNET.modelAE import *
import torch
from torch.optim import SGD, Adam




import os
class IM_NET_Encoder(BaseEncoder):
    def __init__(self):
        args = argparse.Namespace()

        # Assign default values to the attributes
        args.epoch = 0
        args.iteration = 0
        args.learning_rate = 0.00005
        args.beta1 = 0.5
        args.dataset = "all_vox256_img"
        args.checkpoint_dir = "./Models/IMNET/checkpoint"
        args.data_dir = "./Models/IMNET/data/all_vox256_img/"
        args.sample_dir = "./Outputs/"
        args.sample_vox_size = 64
        args.train = False
        args.start = 0
        args.end = 16
        args.ae = True
        args.svr = False
        args.getz = False


        self.modelAE = IM_AE(args)
        self.args=args
        pass
    def GenerateZ(self):
        """
        The function takes input path as the path to the directory where the dataset is stored and generates z-values of the elements in dataset.
        """
        os.system("cd Models/IMNET && python main.py --ae --getz && cd ..  && cd ..")
	    # im_ae = IM_AE(FLAGS)

        return
    def find_optimal_input(self, target_output, initial_input, optimizer_type='SGD', learning_rate=0.01, num_steps=1000):
        model = self.modelAE
        input_var = torch.autograd.Variable(initial_input, requires_grad=True)
        
        if optimizer_type == 'SGD':
            optimizer = SGD([input_var], lr=learning_rate)
        elif optimizer_type == 'Adam':
            optimizer = Adam([input_var], lr=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        for step in range(num_steps):
            optimizer.zero_grad()
            
            output = model(input_var)
            loss = torch.mean((output - target_output) ** 2)  # Mean squared error loss
            
            loss.backward()
            optimizer.step()
            
            if (step + 1) % 100 == 0:
                print(f"Step [{step+1}/{num_steps}], Loss: {loss.item():.4f}")
        
        return input_var.data
    
    
class IM_NET_Decoder(BaseDecoder):
    def __init__(self,model,weight_path):
        args = argparse.Namespace()

        # Assign default values to the attributes
        args.epoch = 0
        args.iteration = 0
        args.learning_rate = 0.00005
        args.beta1 = 0.5
        args.dataset = "all_vox256_img"
        args.checkpoint_dir = "./Models/IMNET/checkpoint"
        args.data_dir = "./Models/IMNET/data/all_vox256_img/"
        args.sample_dir = "./Outputs/"
        args.sample_vox_size = 64
        args.train = False
        args.start = 0
        args.end = 16
        args.ae = True
        args.svr = False
        args.getz = False


        self.modelAE = IM_AE(args)
        self.args=args
        
    def Z2Out(self,z):
        self.modelAE.test_z(self.args,z,256)
        return

