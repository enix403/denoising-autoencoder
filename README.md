# Denoising Images Using Convolutional Autoencoders on CIFAR-10

This project demonstrates how to denoise images using Convolutional Autoencoders, implemented in PyTorch. The CIFAR-10 dataset is used to train and evaluate the model. By adding noise to the images and training the network to reconstruct the original images, the model learns to effectively remove noise.

## Introduction
Image denoising is a critical task in computer vision, where the goal is to recover clean images from noisy inputs. This project uses a Convolutional Autoencoder to denoise images from the CIFAR-10 dataset. Autoencoders are neural networks designed to learn a compressed representation of data (encoder) and reconstruct the data (decoder).

## Dataset
CIFAR-10 is a dataset of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The dataset is divided into 50,000 training and 10,000 testing images.

**Preprocessing:**
- Images are converted to tensors using `torchvision.transforms.ToTensor`.
- Gaussian noise is added to the images during training to simulate noisy inputs.

## Model Architecture
The model consists of two primary components:

### Encoder:
- **Conv2d Layer 1:** Input channels = 3, Output channels = 64, Kernel size = 3, Stride = 2, Padding = 1
- **Conv2d Layer 2:** Input channels = 64, Output channels = 128, Kernel size = 3, Stride = 2, Padding = 1
- **Conv2d Layer 3:** Input channels = 128, Output channels = 256, Kernel size = 3, Stride = 2, Padding = 1
- Each layer is followed by ReLU activation.

### Decoder:
- **ConvTranspose2d Layer 1:** Input channels = 256, Output channels = 128, Kernel size = 3, Stride = 2, Padding = 1, Output Padding = 1
- **ConvTranspose2d Layer 2:** Input channels = 128, Output channels = 64, Kernel size = 3, Stride = 2, Padding = 1, Output Padding = 1
- **ConvTranspose2d Layer 3:** Input channels = 64, Output channels = 3, Kernel size = 3, Stride = 2, Padding = 1, Output Padding = 1
- Each layer is followed by ReLU activation, except the last layer, which uses a Sigmoid activation to output pixel values between 0 and 1.

## Training
- **Loss Function:** Mean Squared Error (MSE) is used to compare the reconstructed image with the original clean image.
- **Optimizer:** Adam optimizer