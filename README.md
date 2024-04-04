# MAE inference for matlab


This folder implements MAE operations. Key components of transformers are located inside `+layers`. To run the code:

First utilize the script inside `weights` folder to convert pytorch model weight to matlab mat file. The url of pytorch model is located at the top of the `convert_pytorch_to_matlab.py` file. The weight should be saved inside the `weights` folder. 

To visualize the attention map, run `MAE_inference_vit_[base/large].m`. This will provide you an interactive program that would give you the attention map. 

To run Masking autoencoder with customized mask, run `MAE_reconstruction_vit_large.m` file. A pop up window will be used to select the masking grids after which MAE inference will be run. The output of the program is a side by side comparison image for original/masked/reconstructed image of the input. 

@Tianqin Li

This program is developed as part of the assignment program in CMU Computer Science class - Neural Computation (Spring 2024)
