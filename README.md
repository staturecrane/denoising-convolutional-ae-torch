# Denoising Convolutional Autoencoder 
An implementation of a denoising convolutional autoencoder built on Torch, trained on still images from Stanley Kubrick's 2001: A Space Odyssey

This code is a modified version of the denoising autoencoder by [Kaixhin](https://github.com/Kaixhin/Autoencoders). It does not have any pooling layers, works on RBG images of 3x96x96 dimensions, and is trained with small, stochastic minibatches, randomly resampling from the full dataset to create a unique training dataset each epoch (as a way of getting around my 4GB GPU limitations). Trained using an NVIDIA 970 GTX. 

#Reconstruction Preview

![](https://github.com/staturecrane/denoising-convolutional-ae-torch/blob/master/2001_reconstruction.gif)
