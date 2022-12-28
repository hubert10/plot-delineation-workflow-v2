# [TF 2.X] Mask R-CNN for Plot Delineation as an application of Object detection and Semantic/Instance Segmentation.

What we are trying to do here is that we take our images and we divide them into overlapping patches, and in that overallaped region, we blend them in a Gaussian way to get smooth predictions.

[Notice] : The original mask-rcnn uses the tensorflow 1.X version. We modified it for tensorflow 2.X version.


### Development Environment

- OS : Ubuntu 20.04.2 LTS
- GPU : Geforce RTX 3090
- CUDA : 11.2
- Tensorflow : 2.2.0 or 2.3.0
- Keras : 2.3.0 or 2.4.0 (tensorflow backend)
- Python 3.8

This is an implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870) on Python 3, Keras, and TensorFlow. The model generates bounding boxes and segmentation masks for each instance of a plot in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.

Mask R-CNN principles

Mask R-CNN is an architecture made of three main parts. First, there is a convolutional network called backbone, which produces features from an input image. From these features, a second part (called RPN for Region Proposal Network) proposes and refines a certain number of regions of interest (as rectangular bounding boxes), which are likely to contain a single cropland. Finally, the last part extracts the best proposals, refines them once again, and produces a segmentation mask for each of them.

Blending smoothing Algorithm:
The following steps summarise the smoothing process to complete the tiles predictions and merging

We did it in the following way:

  1. Original image of size divisible by 1024 is duplicated 8 times, in order to have all the possible rotations and mirrors of that image that fits the possible 90 degrees rotations.
  2. All produced rotations are padded.
  3. Split into tiles (each padded rotated image is split into tiles and predictions are made on every single rotated image)
  4. We perform predictions on each tile.
  5. Combine predictions back into the original size.
  6. Crop padding areas.
  7. Prediction of the rotated image is rotated back to the original orientation.
  8. Results of the both prediction pipelines averaged with geometric mean.

# Step by Step Detection

## 1. Training

To help with training, debugging and understanding the model, there are 2 ways of completing the training process: 

Download the pre-trained weights inside the Mask_RCNN directory. [The weights can be downloaded from this link](https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5)

## 1.1. Use the Colab GPU

[train_model_v2.ipynb](notebooks/train_model_v2.ipynb)

Please, note that you will have to restart runtime to be able 
to use the tensorflow (2.2.0) and keras (2.3.0), otherwise errors will be popped up!

And if everything works well, you will see similar logs as below:

![Logging the training of the model](/photos/training_logs.png)

## 1.2. Run as a python script

You can chose to train the model by running a python script in your terminal:

```
#  Training the model

python train_model.py 
```

In summary, to train the model on your own dataset you'll need to extend the Dataset class:


```Dataset```

This class provides a consistent way to work with any dataset. 

It allows you to use new datasets for training without having to change  the code of the model. It also supports loading multiple datasets at the same time, which is useful if the objects you want to detect are not all available in one dataset [SRV annotated plots](https://drive.google.com/drive/folders/1DljtqDWoBO_-0V6eiYoSvYXkqk9W4Ift?usp=share_link)

These plots were manually annotated using [makesense.ai]: https://www.makesense.ai/ see below: 

![Annotating training images](/photos/annotating_training_images.png)


## Installation

1. Clone this repository 

```bash
git clone https://gitlab.com/rs-working-group/plot-delineation-workflow.git
```

2. Install dependencies

* The versions of Tensorflow and Keras which work well can be installed
with conda package:

```bash
  - conda create -n "condavenv-3.8" python=3.8.0
  - conda activate condavenv-3.8
  - pip3 install -r requirements/requirements.txt
```

We create a virtualenv with python and we name it solaris

```bash
  - sudo apt-get install libgdal-dev
  - sudo apt install libspatialindex-dev python3-rtree
  - python -m venv solaris 
  - source solaris/bin/activate
  - pip3 install -r requirements/solaris_requirements.txt 

```

3. Download pre-trained plots weights (mask_rcnn_object_0030.h5) from the [Google Drive](https://drive.google.com/drive/folders/1j0pBb4j5wtmO-hfiUHqEZDeQFWDhWk9b?usp=share_link).


## 2. Prediction

### 1.1. Make prediction with smooth blending algorithm

One challenge of using a Mask-RCNN for image segmentation is to have smooth predictions, especially if the receptive field of the neural network is a small amount of pixels (case of smallholder farmersfields). 

To overcome this challenge, we decided to use Smooth Blending Algorithm [Smooth Blending Algorithm](https://github.com/Vooban/Smoothly-Blend-Image-Patches) which aims at performing smooth predictions on an image from tiled prediction patches. The main steps of this algorithm can be summarized in these steps: 

  1. Original (3600, 3600) image is rotated by 90 degrees and we get 2 images: original and rotated.
  2. Both are padded.
  3. Split into tiles(each padded rotation image is split into tiles and predictions are done on it)
  4. We perform predictions on each tile.
  5. Combine predictions back into the original size.
  6. Crop padding areas.
  7. Prediction of the rotated image is rotated back to the original orientation.
  8. Results of the both prediction pipelines averaged with geometric mean.

```
# Run the prediction on the trained model

python make_smooth_predictions.py 
```

## 3. Post-preprocessing

The predictions are a gray scale image which requires more post-preprocessing to have good predicited masks.

### 1.1. Image Filtering

To have distinct and transparent pixels, we apply a filter by cleaning the grayscale image to whiten the pixels inside the plots and darken the other parts (outside of plot boundaries).

```
# Run the prediction on the trained model

python postpreprocessing/1_image_filtering.py 
```

## 1.2. Geo-referencing

The predicted masks do not contain any metadata at this level, so we need to convert prediction patches saved in jpg, png, etc to TIF format referencing the metadata of the original satelite imagery

```
# Run the prediction on the trained model

python postpreprocessing/2_geo_referencing.py 
```

## 1.3. Solaris Visualization

The below image shows the predicted polygones using the Smooth Blending Algorithms:

![Predictions with Smooth Blending](/photos/predictions_with_smooth_blending.png)

<!-- <img src="https://gitlab.com/rs-working-group/plot-delineation-workflow/blob/main/photos/predictions_with_smooth_blending.png" width="520"/>  -->


```
# Visualize the prediction using Solaris 

python visualization/4_solaris_predictions_viz.py 
```


## 1.4. Plot size distribution

After converting predicted raster into vector layers, we compare the distribution of the predicted
polygones against the initial farmers plots


```
# Visualize the extracted polygones alongside ground truth plot boundaries 

python visualization/5_plot_size_distribution.py 
```

