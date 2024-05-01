import torch
from utils import *
class BaseEncoder:
    def __init__(self,model,weight_path):
        self.model=model
        self.weight_pth=weight_path
        self.model.load_state_dict(torch.load(self.weight_pth))
        
    def GenerateZ(self,input):
        z = self.model(input)
        ## Save as hdf5 format as key "zs"
        write_hdf5("data\Z-Embeddings\Z_base.hdf5","zs",z.detach().cpu().numpy())
        return
        

