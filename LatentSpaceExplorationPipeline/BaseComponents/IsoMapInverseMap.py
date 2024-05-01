from utils import *
import numpy as np
from PIL import Image, ImageDraw
class Mapper:
    def __init__(self):
        self.latent_img=None
        self.embedding=None
        self.inverse_coeff=None
        return
    def computeIsoInvMap(self,z_filepath,number=400,neighbors_num = 7,reduced_n = 2):
        if(number is None):
            z_latents=read_hdf5(filename=z_filepath)
        else:
            z_latents=read_hdf5(filename=z_filepath)[:number]
        iso_z_2,embedding = isomapping(z_latents, neighbors_num, reduced_n)
        width = 500
        height = 500
        image = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(image)
        color = (0, 255, 0)
        point_size=5
        min_z=np.min(iso_z_2,axis=0)
        max_z=np.max(iso_z_2,axis=0)
        self.min_z=min_z
        self.max_z=max_z

        for point in iso_z_2:
            x, y = point
            x=(x-min_z[0])/(max_z[0]-min_z[0])*500
            y=(y-min_z[1])/(max_z[1]-min_z[1])*500

            # draw.point((x, y), fill=color)
            draw.rectangle([(x, y), (x+point_size, y+point_size)], fill=color)

        self.latent_img=image
        self.embedding = embedding
        self.inverse_coeff = inverse_map(z_latents,iso_z_2)
       
        return
    
    def computeInverse(self,z_2):
        
        return invert(z_2,self.inverse_coeff)
        