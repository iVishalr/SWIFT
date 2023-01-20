# Towards Faster and Efficient Lightweight Image Super Resolution using Swin Transformers and Fourier Convolutions

[Vishal Ramesha](https://github.com/iVishalr), [Abhishek Aditya BS](https://github.com/Abhishek-Aditya-bs), [Yashas Kadambi](https://github.com/Yashas120), [T Vijay Prashant](https://github.com/tvijayprashant), Shylaja S S

This repository contains the official implementation for the paper titled "Towards Faster and Efficient Lightweight Image Super Resolution using Swin Transformers and Fourier Convolutions" presented at NTIRE Workshop, CVPR 2023.

## Image Super-Resolution

Image Super Resolution (SR) is the process of recovering or generating high-resolution(HR) images from low-resolution (LR) images. With the advancement in deep learning techniques in recent years, deep learning-based SR models have been actively explored and often achieve state-of-the-art performance on various benchmarks of SR. A variety of deep learning methods have been applied to solve SR tasks, ranging from the early Convolutional Neural Networks (CNN) based method to recent promising Generative Adversarial Networks and transformer based SR approaches.

## Problem Statement

Generation of high quality images with good results, free of artefacts and odd textures from low resolution images is our main goal.

To make our project work in real time, we plan to make use of efficient architectures that can operate in environments where there are limits on the computation and memory available for running the models. 

## Scope of the Project

Image super resolution involves taking a low resolution blurry image as input and generating a high resolution image while maintaining the context of the input low resolution image. It is popularly used in the following applications :

* Surveillance: to detect, identify, and perform facial recognition on low-resolution images obtained from security cameras.
* Medical: capturing high-resolution MRI images can be tricky when it comes to scan time, spatial coverage, and signal-to-noise ratio (SNR). Super resolution helps resolve this by generating high-resolution MRI from otherwise low-resolution MRI images.
* Media: super resolution can be used to reduce server costs, as media can be sent at a lower resolution and upscaled on the fly.

## Objectives

* Upsample low resolution images by 2x, 3x and 4x scales.
* Efficient model inference for real time applications.
* Reduction in model footprint by reducing the number of parameters.
* Reduction in artifacts introduced during upsampling.
* Model should be able to work with any type of input images.
## Model Architecture

<p align="center">
<img src="Images-Store/model-arch.png" alt="model-arch" style="zoom:50%;" />
</p>

The model consists of three main modules, Shallow Feature Extraction to extract the Low frequency details from the training examples, Deep Feature Extraction which is responsible for learning different patterns and structures and to further extract finer high frequency details in the images and High Quality Image Reconstruction Module which upscales the images to the desired size with no artifacts and other image distortions. A single skip connection is added from the shallow feature module to the deep feature module to tackle the vanishing gradients problem and to enhance the quality of the output produced.

## Dataset and Metrics

To train our models, DIV2K and Flickr2K datasets have been used. DF2K dataset is generated by merging both DIV2K and Flickr2K datasets. For evaluation of the model, The five benchmark datasets - BSD100, Manga109, Urban100 Set5 and Set14 are used. For quantitative evaluation PSNR and SSIM metrics are used to measure the quality of reconstructed SR images. 

## Training

Randomly crop LR images of size 64 x 64 are passed as inputs in each epoch. Data augmentation techniques like random horizontal flips, vertical flips and RGB Channel Shuffling are used. Model was trained using a learning rate of 2e-4 initially and then was decreased by half at [200000, 400000, 450000, 475000] iterations. Adam optimizer was used for training with (0.9, 0.99) as beta values. We used the PyTorch framework to build our models and all the training was performed on NVIDIA TESLA A100 80GB GPUs on Google Cloud Platform.

### Results

<p align="center">
<img src="Images-Store/results.png" alt="Results" style="zoom:50%;" />
</p>

## Usage
### Benchamrk Testing

1. Clone the repository

   ```bash
   git clone https://github.com/Image-SuperResolution/EffcNet-Efficient-Fast-Fourier-Convolution-Network
    ```
2. Install the Dependencies (Use either Pip or Conda)

   ```bash
    conda create -n pytorch
    conda activate pytorch
    conda install mamba -n base -c conda-forge
    mamba install numpy pandas matplotlib tqdm
    mamba install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
    mamba install -c conda-forge opencv scikit-image timm einops
    pip3 install heic2png
   ```

3. The pretrained models are stored in `experiment` folder. Change the path of the pretrained model in `TEST.py` file on line 32.

```python
    parser.add_argument("--model_path", 
    default="/Users/abhishekadityabs/Desktop/EffcNet/experiment/4x-674k.pth", 
    type=str, help="path to trained model")
```

4. Change the path of the datasets in `TEST.py` file on line 58

   ```bash
    testset_personal = prepare_testset.MakeTestSet("/Users/abhishekadityabs/Desktop/EffcNet/TESTSETS/personal/",None,args.scale,args.patch_size,args.OOM,True)
   ```
   Similarly change the paths if you want to test on other datasets.
   
   **You can use your own photos and store them in the `personal` folder and test on them.**

4. For scale x4 use the following arguments

   ```bash
    parser.add_argument("--scale", type=int, default=4,
                    help="scale")
    parser.add_argument("--patch_size", type=int, default=256,
                    help="patch size")
   ```
    For scale x3 use the following arguments
    
    ```bash
     parser.add_argument("--scale", type=int, default=3,
                      help="scale")
     parser.add_argument("--patch_size", type=int, default=192,
                      help="patch size")
    ```

5. Run the `TEST.py` file

   ```bash
    python3 TEST.py
   ```
    The results will be stored in the `Results` folder.
## Future Work

* Publish the paper in Top Conferences.
* Develop a Web Interface for the end users to use the application.
* Give a shot at Video Super-Resolution.
* Train a larger model, Classical Model, for x2, x3, x4 scales.
* Optimize for Mobile usage using PyTorch Mobile support.

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Authors

* [@Abhishek Aditya](https://github.com/Abhishek-Aditya-bs)
* [@T Vijay Prashant](https://github.com/tvijayprashant)
* [@Vishal R](https://github.com/iVishalr)
* [@Yashas KS](https://github.com/Yashas120)
