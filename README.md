# Mobile-CoHAtNet: Lightweight Hybrid Convolution-Transformer for Camera Localization Using RGB, Depth and IMU data

Mobile-CoHAtNet is a lightweight hybrid convolution-transformer architecture optimized for efficient and accurate 6-DoF camera localization. It integrates RGB, depth, and IMU data, making it suitable for resource-constrained devices such as mobile and embedded systems.


---

## Overview

Mobile-CoHAtNet leverages the strengths of:
- **Depthwise Separable Convolutions**: For efficient feature extraction.
- **MBConv Blocks**: To capture fine-grained local details.
- **Hybrid Self-Attention Mechanisms**: For modeling global spatial and detailed fine grained relationships.
- **IMU Data Integration**: Generated from transformation matrices, enhancing robustness in visually challenging environments.

![Mobile-CoHAtNet](https://github.com/Husseinhhameed/Mobile-CoHAtNet/blob/main/Concept.png)


![IMUt](https://github.com/Husseinhhameed/Mobile-CoHAtNet/blob/main/IMUfusing.png)


This repository includes:
- Scripts for Creating and training Mobile-CoHAtNet.
- Script of Estimating IMU data from transformation matrix.
- LAB dataset recorded using the **Starry Scanner App** on an iPhone.

## Running on Google Colab

### Train Mobile-CoHAtNet
1. Open `Mobile_CoHAtnet.ipynb` in Google Colab.
2. Follow the instructions to load your dataset from Google Drive and start training.

### Generate IMU Data
1. Open `estimate_IMU_from_Transformation_matrix.ipynb` in Google Colab.
2. Provide transformation matrices as input to simulate IMU data for your dataset.

## LAB Dataset Details

The **LAB dataset** was captured using the [**Stray Scanner App**](https://docs.strayrobots.io/)  on an iPhone, leveraging its LiDAR sensor for synchronized RGB and depth images. This dataset includes:

- **RGB images**: High-resolution images captured from various angles.
- **Depth maps**: Providing precise depth information for each scene.
- **Simulated IMU data**: Extracted from transformation matrices to enhance localization performance.
- **Ground truth pose values**: Accurate 6-DoF poses for each image.

The dataset represents a laboratory environment with challenging conditions such as:
- Cluttered areas with various objects.
- Texture-less surfaces that make visual localization harder.
- Repetitive patterns that may introduce ambiguity in pose estimation.

You can download the dataset from the following Google Drive link:  
[Download LAB Dataset](https://drive.google.com/file/d/1voslJg1x0EB8Fck0Xuodc1Cf0mINvknq/view?usp=sharing)

![Dataset]([https://github.com/Husseinhhameed/Mobile-CoHAtNet/blob/main/IMUfusing.png](https://github.com/Husseinhhameed/Mobile-CoHAtNet/blob/main/Dataset.png))



