import torch
class BaseDecoder:
    def __init__(self,model,weight_path):
        self.model=model
        self.weight_pth=weight_path
        self.model.load_state_dict(torch.load(self.weight_pth))
        
    def Z2Out(self,z):
        outputs = self.model(z)
        ## Process the output and return it in the wrapper
        return outputs