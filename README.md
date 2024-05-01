## ALICE : Artificial Latent-space Interactive Exploration
##### Welcome to ALICE (Artificial Latent-space Interactive Exploration)! ALICE is a tool that allows you to explore the latent space of a generative model       interactively. You can adjust latent vectors and visualize the corresponding outputs in 3D. Choose your model configuration and mode of operation below to get started.

You can find the demo over here
```

```

To run the GUI locally
```
# clone the repo
git clone https://github.com/Atharvap14/ALICE.git
cd ALICE
```

*  Download IMNET folder from the [here](https://drive.google.com/drive/folders/1OxNWDbgBL4bdo33xuIGhZofTbJOUhvy5?usp=sharing) and place it in "LatentSpaceExplorationPipeline\Models"

```
conda env create -f LaTexP_environment.yml
conda activate LaTexP
pip install -r requirements.txt
cd LatentSpaceExplorationPipeline
streamlit run main.py
```

*  video can be found here
   *  [demo](https://drive.google.com/file/d/1YbYlDC7IUYdYEQgg8Lxk34M1sVtvUoQ8/view?usp=sharing)
   *  [introduction](https://drive.google.com/file/d/1_6kvZImpU3Y3hnUd-r16Bu3Zr_bVC8Az/view?usp=sharing)