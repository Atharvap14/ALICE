import streamlit as st
import plotly.graph_objects as go
from utils import *
import numpy as np
import argparse
from Wrappers.IMNETWrapper import *
import torch
from streamlit_image_coordinates import streamlit_image_coordinates
import os
from pywavefront import Wavefront
from plotly.subplots import make_subplots
import open3d as o3d
from BaseComponents.IsoMapInverseMap import *
import time
import requests
from PIL import Image, ImageDraw
from io import BytesIO
class GUI:
    
    def __init__(self):
        
        st.set_page_config(layout='wide')  # Enable wide layout
        st.title("ALICE : Artificial Latent-space Interactive Exploration") 
        st.markdown("""                 
        Welcome to ALICE (Artificial Latent-space Interactive Exploration)! ALICE is a tool that allows you to explore the latent space of a generative model 
        interactively. You can adjust latent vectors and visualize the corresponding outputs in 3D. Choose your model configuration and mode of operation below to get started.
        \n(Help is offered here) [ℹ️](Hover-over-these-through-out-the-GUI-to-get-more-info.)""", unsafe_allow_html=True)
        
        
        
        with st.expander("Model Config [ℹ️](Define-model-configuration.)"):
            self.model_config()
        
        with st.expander("Mode of Operation [ℹ️](Hover-over-these-through-out-the-GUI-to-get-more-info.)"):
            self.mode_of_operation()
        self.dynamic_interaction()
    def p_view1(self):
        st.header("")
        
        col1, col2 = st.columns(2)


        with col1:
            st.markdown("# Latent Space")
            st.markdown("Select latent vector to brew")
            ## The following is the input system for the interface
            
            image = self.map.latent_img

            

            # Create a canvas for drawing
            draw_canvas = st.empty()

            # Create a copy of the image to draw on
            draw_image = image.copy()

            # Create a drawing object
            draw = ImageDraw.Draw(draw_image)
            if 'points'  in st.session_state:
                    point = st.session_state["points"][-1]
                    x,y=point
                    draw.rectangle([x-2, y-2, x+2, y+2], outline='red', width=3)


            value = streamlit_image_coordinates(draw_image, key="pil")
            if value is None:
                value={"x":250,"y":250}
            if value is not None:
                point = value["x"], value["y"]
                if 'points'  in st.session_state:
                    if point not in st.session_state["points"]:
                        st.session_state["points"].append(point)
                        st.experimental_rerun()
                else:
                    st.session_state["points"]=[point]
                    st.experimental_rerun()
            
            

        with col2:
            st.markdown("# Output 3D Model")
            st.markdown("Your Tea ☕︎ Served Here (use mouse to interact)")
            name=r"Outputs\out0.ply"
            inp=np.array([[value["x"],value["y"]]])
            inp[0][0]=inp[0][0]/500*(self.map.max_z[0]-self.map.min_z[0])+self.map.min_z[0]
            inp[0][1]=inp[0][1]/500*(self.map.max_z[1]-self.map.min_z[1])+self.map.min_z[1]
            
            # print(inp)
            sttime = time.time()
            self.inference( self.map.computeInverse(inp))
            it = time.time()
            print("Inference time = ",it-sttime)
            st.plotly_chart(self.parse_ply_file(name))
            print("Plotting time = ",time.time()-it)
            print("Overall time =",time.time()-sttime)

    def p_view2(self):
        st.header("")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("# Latent Space")
            st.markdown("Select latent vector to cook")
            ## The following is the input system for the interface
            min_value = 0
            max_value = 100
            default_value = 50
            image = self.map.latent_img

            

            # Create a canvas for drawing
            draw_canvas = st.empty()

            # Create a copy of the image to draw on
            draw_image = image.copy()

            # Create a drawing object
            draw = ImageDraw.Draw(draw_image)
            if 'lines'  in st.session_state:
                    point1 = st.session_state["lines"][-2]
                    point2 = st.session_state["lines"][-1]

                    draw.line([point1,point2], fill='black', width=2)
                    
                    try:
                        selected_value = st.session_state["sv"]
                        x = point1[0]*(1-(selected_value/max_value))+point2[0]*(selected_value/max_value)
                        y = point1[1]*(1-(selected_value/max_value))+point2[1]*(selected_value/max_value)
                    except:
                        print("Ran Exception")
                        x = point1[0]*(1-(default_value/max_value))+point2[0]*(default_value/max_value)
                        y = point1[1]*(1-(default_value/max_value))+point2[1]*(default_value/max_value)

                    draw.rectangle([x-2, y-2, x+2, y+2], outline='red', width=3)

            


            value = streamlit_image_coordinates(draw_image, key="pil")
            

            # Create the slider widget
            selected_value = st.slider('Select a value:', min_value, max_value, default_value)
            if('sv'  in st.session_state):
                if(selected_value!=st.session_state["sv"]):
                    st.session_state["sv"]=selected_value
                    st.experimental_rerun()
            else:
                 st.session_state["sv"]=selected_value
                 st.experimental_rerun()

            if value is None:
                value={"x":250,"y":250}
            if value is not None:
                point = value["x"], value["y"]
                if 'lines'  in st.session_state:
                    if point not in st.session_state["lines"]:
                        st.session_state["lines"].append(point)
                        st.experimental_rerun()
                else:
                    st.session_state["lines"]=[point,point]
                    st.experimental_rerun()
            
            

        with col2:
            st.markdown("# Output 3D Model")
            st.markdown("Your Tea ☕︎ Served Here")
            if st.button("Brew ☕︎"):
                name=r"Outputs\out0.ply"
                if selected_value is None:
                            x = point1[0]*(1-(default_value/max_value))+point2[0]*(default_value/max_value)
                            y = point1[1]*(1-(default_value/max_value))+point2[1]*(default_value/max_value)
                else:
                            x = point1[0]*(1-(selected_value/max_value))+point2[0]*(selected_value/max_value)
                            y = point1[1]*(1-(selected_value/max_value))+point2[1]*(selected_value/max_value)
                inp=np.array([[x,y]])
                inp[0][0]=inp[0][0]/500*(self.map.max_z[0]-self.map.min_z[0])+self.map.min_z[0]
                inp[0][1]=inp[0][1]/500*(self.map.max_z[1]-self.map.min_z[1])+self.map.min_z[1]
                
                # print(inp)
                self.inference( self.map.computeInverse(inp))
                st.plotly_chart(self.parse_ply_file(name))
    @st.cache_resource
    def load_model(_self,model,weights,input_path):
        ## TO-DO: This is a basic demo for directly plugging the Implicit decoder in the pipeline. Generalize it!!
        # Reading Training Latents
        if(model=="IMNET Decoder" and weights=="Default" and input_path=="Default-Embedding"):
            
            filename = r"Models\IMNET\checkpoint\all_vox256_img_ae_64\all_vox256_img_train_z.hdf5"
            
            _self.map=Mapper()
            _self.map.computeIsoInvMap(filename)
            print("Map generated")
            _self.decoder= IM_NET_Decoder(model,weights)
            return _self.decoder, _self.map
    

    def model_config(self):
        st.header("Model Config")
        self.model_name = st.selectbox("Model Name", ["IMNET Decoder"])
        self.weights_file = st.selectbox("Weights_File", ["Default"])
        self.input_file = st.selectbox("InputFile", ["Default-Embedding"])
        

    
        self.decoder,self.map = self.load_model(self.model_name, self.weights_file, self.input_file)
        st.success("Model loaded successfully!")

                

        # self.decoder,self.map=self.load_model(None,None,None)

    def mode_of_operation(self):
        st.header("Mode of Operation")
        self.mode = st.radio("Select Mode", ["Pointex Mode: To explore latent space pointwise", "Linex Mode: To interpolate between two latent points"])
        # self.mode = st.radio("Select Mode", ["Pointex Mode", "Linex Mode"])

        # Perform further processing based on the selected mode

    def dynamic_interaction(self):
        st.header("Welcome to WonderLand's Tea Party")

        if self.mode == "Pointex Mode: To explore latent space pointwise":
            self.p_view1()  # Call function to render UI for Pointex Mode
        elif self.mode == "Linex Mode: To interpolate between two latent points":
            self.p_view2()  # Call function to render UI for Linex Mode

    def inference(self,z):
        # print(z)
        self.decoder.Z2Out(z)

    def parse_obj_file(self,obj_file_path):
        obj = Wavefront(obj_file_path, collect_faces=True)
        vertices = np.array(obj.vertices)
        faces = np.array(obj.mesh_list[0].faces)
        # Create a list of x, y, and z coordinates from the vertices
        x_coords = vertices[:, 0]
        y_coords = vertices[:, 1]
        z_coords = vertices[:, 2]

        # Create the mesh object using the vertex and face data
        mesh = go.Mesh3d(x=x_coords, y=y_coords, z=z_coords, i=faces[:, 0], j=faces[:, 1], k=faces[:, 2])
        # x_min, x_max = min(mesh.x), max(mesh.x)
        # y_min, y_max = min(mesh.y), max(mesh.y)
        # z_min, z_max = min(mesh.z), max(mesh.z)
        
        fig = go.Figure([mesh])
        return fig

    
    def parse_ply_file(self, ply_file_path):
        mesh = o3d.io.read_triangle_mesh(ply_file_path)
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)

        # Create a list of x, y, and z coordinates from the vertices
        x_coords = vertices[:, 0]
        y_coords = vertices[:, 1]
        z_coords = vertices[:, 2]

        # Create the mesh object using the vertex and face data
        mesh = go.Mesh3d(x=x_coords, y=y_coords, z=z_coords, i=faces[:, 0], j=faces[:, 1], k=faces[:, 2])
        # x_min, x_max = min(mesh.x), max(mesh.x)
        # y_min, y_max = min(mesh.y), max(mesh.y)
        # z_min, z_max = min(mesh.z), max(mesh.z)
        # Create the Plotly figure
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(mesh)
        # Adjust the scene layout based on the mesh size
        # glob_min=max(0,min(x_min,y_min,z_min)-0.2)
        # glob_max=max(x_max,y_max,z_max)+0.5
        fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        ),
        scene_aspectmode='data',
        showlegend=False
    )
        return fig



    

    
