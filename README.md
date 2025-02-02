# <h1 align="center">ECE 046211 - Technion - Deep Learning - Project </h1> 
## <h2 align="center"> "Mind the Gap" - A Deep Learning Analysis of Pothole Detection </h2>

<h4 align="center">
  <table align="center">
    <tr>
      <td align="center">
        <img src="./data/readme/Itai.png" width="100" height="100"/> <br>
        <strong>Itai Benyamin</strong> <br>
        <a href="https://www.linkedin.com/in/itai-benyamin/">
          <img src="./data/readme/LinkedInLogo.png" width="50" height="50"/>
        </a>
        <a href="https://github.com/Itai-b">
          <img src="./data/readme/GitHubLogo.png" width="50" height="50"/>
        </a>
      </td>
      <td align="center">
        <img src="./data/readme/Idan.png" width="100" height="100"/> <br>
        <strong>Idan Baruch</strong> <br>
        <a href="https://www.linkedin.com/in/idan-baruch-76490a181/">
          <img src="./data/readme/LinkedInLogo.png" width="50" height="50"/>
        </a>
        <a href="https://github.com/idanbaru">
          <img src="./data/readme/GitHubLogo.png" width="50" height="50"/>
        </a>
      </td>
    </tr>
  </table>
</h4>

## Abstract

This project focuses on the detection and classification of potholes in road infrastructure, with a specific emphasis on categorizing them by their severity levels. The goal is to develop a deep learning-based solution capable of accurately identifying potholes and determining their severity from images.

We trained different state-of-the-art (SOTA) object detection models for this task and evaluated their performance under various conditions, including self-synthesized motion blur noise. This noise was generated using randomized kernel types and sizes to simulate real-world conditions where camera shake or motion artifacts may degrade image quality. The study provides insights into how motion blur affects detection accuracy and offers recommendations for improving the robustness of automated road maintenance systems.

## Table of Contents

## Files in the repository

| File Name                       | Purpose                                                                 |
|---------------------------------|-------------------------------------------------------------------------|
| `data_process.py`               | Handles data preprocessing, such as loading and transforming datasets.  |
| `trainer.py`                    | Manages the training process for torchvision models.                    |
| `evaluator.py`                  | Evaluates the performance of trained models using metrics.              |
| `motion_blur.py`                | Applies our self synthesized motion blur effects to images for testing. |
| `utils.py`                      | Contains utility functions used across different modules.               |
| `torchvision_models_train.ipynb`| Jupyter notebook for training models using torchvision.                 |
| `models_data.json`              | Stores metadata for the models results                                  |
| `main.ipynb`                    | Main notebook for orchestrating the project workflow.                   |
| `data`                          | Directory containing all the datasets and results                       |
| `HW`                            | Directory for our homework assigments.                                  |

## Installation Instructions

For the complete guide, with step-by-step images, please consult `Setting Up The Working Environment.pdf`

1. Get Anaconda with Python 3, follow the instructions according to your OS (Windows/Mac/Linux) at: https://www.anaconda.com/download
2. Install the basic packages using the provided `environment.yml` file by running: `conda env create -f config/environment.yml` which will create a new conda environment named `deep_learn`. You can use `config/environment_no_cuda.yml` for an environment that uses pytorch cpu version.
3. Alternatively, you can create a new environment and install packages from scratch:
In Windows open `Anaconda Prompt` from the start menu, in Mac/Linux open the terminal and run `conda create --name deep_learn`. Full guide at https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands
4. To activate the environment, open the terminal (or `Anaconda Prompt` in Windows) and run `conda activate deep_learn`
5. Install the required libraries according to the table below (to search for a specific library and the corresponding command you can also look at https://anaconda.org/)

### Libraries to Install

|Library           | Command to Run |
|------------------|-----------------------------------------|
|`Jupyter Notebook`|  `conda install -c conda-forge notebook`|
|`numpy`           |  `conda install -c conda-forge numpy`|
|`matplotlib`      |  `conda install -c conda-forge matplotlib`|
|`pandas`          |  `conda install -c conda-forge pandas`|
|`scipy`           | `conda install -c anaconda scipy `|
|`scikit-learn`    |  `conda install -c conda-forge scikit-learn`|
|`seaborn`         |  `conda install -c conda-forge seaborn`|
|`tqdm`            | `conda install -c conda-forge tqdm`|
|`opencv`          | `conda install -c conda-forge opencv`|
|`optuna`          | `pip install optuna`|
|`optuna-dashboard`| `pip install optuna-dashboard`|
|`kagglehub`       | `pip install kagglehub` |
|`kornia`          | `pip install kornia` |
|`xmltodict`       | `pip install xmltodict`|
|`torchmetrics`    | `pip install torchmetrics` |
|`ipywidgets`      | `pip install ipywidgets`|
|`ultralytics`     | `conda install -c conda-forge ultralytics`|
|`pytorch` (cpu)   | `conda install pytorch torchvision cpuonly -c pytorch` (<a href="https://pytorch.org/get-started/locally/">get command from PyTorch.org</a>)|
|`pytorch` (gpu)   | `conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia` (<a href="https://pytorch.org/get-started/locally/">get command from PyTorch.org</a>)|

## Dataset
We used the following dataset from kaggle [chitholian_annotated_potholes_dataset](https://www.kaggle.com/datasets/chitholian/annotated-potholes-dataset)

The Dataset contains 665 images of potholes on roads with corresponding annotations boxes for each pothole.

We created a custom torch dataset named PotholeDetectionDataset and then split them to seperate train, validation and test sets (70-10-20).

<div align="center">
  <img src="./data/plots/random_images_from_train.png"/>
</div>

## Object Detection Models
In this project we trained the following SOTA object detection models:

[torchvision models](https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection):
- `ssd`
- `faster rcnn`
- `retinanet`
- `fcos`

[ultralytics yolo](https://docs.ultralytics.com/models/yolov8/):
- `yolov8m`

### Hyperparameter Tuning
In this project, hyperparameter tuning was conducted using Optuna to optimize training on various torchvision models.
A comprehensive search was performed across multiple hyperparameter spaces, including model preweight strategies, optimizer types, learning rates, momentum, and weight decay parameters.
The study involved carefully selecting combinations of hyperparameters to achieve the best validation mean average precision (mAP).
The objective function was optimized using `MedianPruner` and `TPESampler` for better exploration and exploitation across trials. Below is a detailed table of the tuned hyperparameters.

| **Category** | **Hyperparameter** | **Range/Choices**                  | **Description**                          |
|--------------|--------------------|--------------------------------------|--------------------------------------------|
| Model        | `preweight_mode`   | `['random', 'freezing', 'fine_tuning']` | Strategy for loading model weights      |
| Training     | `batch_size`       | `[4, 8]`                            | Number of samples per training batch      |
| Training     | `epochs`           | `[10, 20]`                          | Number of training epochs                 |
| Optimizer    | `optimizer`        | `['SGD', 'Adam', 'AdamW', 'RMSprop']` | Optimization algorithm                   |
| SGD          | `lr`               | `[5e-3, 5e-2]` (log scale)          | Learning rate for SGD                     |
| SGD          | `momentum`         | `[0.9, 0.99]`                       | Momentum for SGD                          |
| SGD          | `weight_decay`     | `[1e-5, 1e-3]` (log scale)          | Weight decay for SGD                      |
| Adam/AdamW   | `lr`               | `[1e-4, 1e-2]` (log scale)          | Learning rate for Adam/AdamW              |
| Adam/AdamW   | `beta1`            | `[0.8, 0.999]`                      | Beta1 parameter                           |
| Adam/AdamW   | `beta2`            | `[0.9, 0.999]`                      | Beta2 parameter                           |
| RMSprop      | `lr`               | `[1e-3, 1e-2]` (log scale)          | Learning rate for RMSprop                 |
| RMSprop      | `momentum`         | `[0.9, 0.99]`                       | Momentum for RMSprop                      |
| RMSprop      | `weight_decay`     | `[1e-1, 1]` (log scale)             | Weight decay for RMSprop                  |
| Scheduler    | `scheduler`        | `['StepLR', 'CosineAnnealingLR', 'ReduceLROnPlateau', 'OneCycleLR']` | Learning rate scheduler type |
| StepLR       | `step_size`        | `[2, 5]`                            | Step size for StepLR                      |
| StepLR       | `gamma`            | `[0.05, 0.5]`                       | Decay factor for StepLR                   |
| CosineAnneal | `T_max`            | `[5, 15]`                           | Maximum cycle length for CosineAnnealingLR|
| CosineAnneal | `eta_min`          | `[1e-7, 1e-5]` (log scale)          | Minimum learning rate                     |
| Plateau      | `factor`           | `[0.1, 0.5]`                        | Decay factor for ReduceLROnPlateau        |
| Plateau      | `patience`         | `[2, 5]`                            | Patience for ReduceLROnPlateau            |
| OneCycleLR   | `max_lr`           | `[1e-4, 1e-2]` (log scale)          | Maximum learning rate for OneCycleLR      |

- The best configurations for each model was saved for future training (i.e. `data/models/fasterrcnn_resnet50_fpn/fasterrcnn_resnet50_fpn_best_params.json`)
- Results available in Optuna dashboard at `./data/models/db.sqlite3`
  - launch it using:
  ```
  optuna-dashboard ./data/models/db.sqlite3
  ```
  - then open your browser at `http://localhost:8080`
- At the end, each configuration was set to be trained with a batch size of 8 for 100 epochs.
- Best weights achieving the highest mAP@50 on the validation set were saved.

## Results

<div align="center">
  <img src="./data/plots/training_loss_val_map.png"/>
</div>

<div align="center">
  <img src="./data/plots/test_map_fps/clean_test_map_fps.png"/>
</div>

## Motion Blur Noise

<div align="center">
  <img src="./data/motion_blur_data/motion_blur_types.png"/>
</div>

## Data Augmentations
To cope with the motion blur noise, we applied the following data augmentations using [kornia](https://kornia.readthedocs.io/en/latest/):

| **Augmentation**       | **Parameters**                                                                 | **Description**                                                                 |
|------------------------|---------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| [`RandomMotionBlur`](https://kornia.readthedocs.io/en/latest/augmentation.module.html#kornia.augmentation.RandomMotionBlur)     | `kernel_size=(3, 51)`, `angle=(-180.0, 180.0)`, `direction=(-1.0, 1.0)`, `p=0.4` | Applies random motion blur to simulate camera shake and motion blur.            |
| [`RandomGaussianBlur`](https://kornia.readthedocs.io/en/latest/augmentation.module.html#kornia.augmentation.RandomGaussianBlur)   | `kernel_size=(3, 3)`, `sigma=(0.1, 2.0)`, `p=0.3`                               | Applies random Gaussian blur to simulate out-of-focus blur.                     |
| [`RandomSharpness`](https://kornia.readthedocs.io/en/latest/augmentation.module.html#kornia.augmentation.RandomSharpness)      | `sharpness=(0.5, 2.0)`, `p=0.3`                                                 | Adjusts the sharpness of the image to simulate varying levels of focus.         |

These augmentations were chosen to help the model generalize better to different types of blur that might be encountered in real-world scenarios.
- The `RandomMotionBlur` simulates the effect of motion blur caused by camera shake or object movement.
- The `RandomGaussianBlur` simulates the effect of out-of-focus blur, which can occur due to incorrect focus settings.
- The `RandomSharpness` augmentation adjusts the sharpness of the image, simulating varying levels of focus and helping the model to learn to handle both sharp and blurry images.

## Results

### Results on the Clean Test Set

<div align="center">
  <img src="./data/plots/test_map_fps/clean_test_map_fps_with_aug.png" style="height: 500px;"/>
</div>

### Results on the Test Set Test Set with Uniform Motion Blur

<div align="center">
  <img src="./data/plots/test_map_fps/uniform_test_map_fps.png" style="height: 250px;"/>
</div>

### Results on the Test Set with Ellipse Motion Blur

<div align="center">
  <img src="./data/plots/test_map_fps/ellipse_test_map_fps.png" style="height: 250px;"/>
</div>

### Results on the Test Set with Natural Motion Blur

<div align="center">
  <img src="./data/plots/test_map_fps/natural_test_map_fps.png" style="height: 500px;"/>
</div>

## Potholes Severity

<div align="center">
  <img src="./data/plots/potholes_with_severity_images.png"/>
</div>

<div align="center">
  <img src="./data/plots/pothole_class_distribution.png"/>
</div>


## References

1. Z. Zou, K. Chen, Z. Shi, Y. Guo, and J. Ye. *Object Detection in 20 Years: A Survey*. Proceedings of the IEEE, vol. 111, no. 3, pp. 257-276, March 2023. DOI: [10.1109/JPROC.2023.3238524](https://doi.org/10.1109/JPROC.2023.3238524).

2. *Object Detection on COCO*. Papers with Code. [Link](https://paperswithcode.com/sota/object-detection-on-coco).

3. *Object Detection Leaderboard*. Hugging Face. [Link](https://huggingface.co/blog/object-detection-leaderboard).

4. *Faster R-CNN vs YOLO vs SSD: Object Detection Algorithms*. Medium, IBM Data & AI. [Link](https://medium.com/ibm-data-ai/faster-r-cnn-vs-yolo-vs-ssd-object-detection-algorithms-18badb0e02dc).

5. Tamagusko, T., Ferreira, A. *Optimizing Pothole Detection in Pavements: A Comparative Analysis of Deep Learning Models*. Eng. Proc. 2023, 36, 11. DOI: [10.3390/engproc2023036011](https://doi.org/10.3390/engproc2023036011).

6. A. Levin, Y. Weiss, F. Durand, and W. T. Freeman. *Understanding and Evaluating Blind Deconvolution Algorithms*. 2009 IEEE Conference on Computer Vision and Pattern Recognition, Miami, FL, USA, 2009, pp. 1964-1971. DOI: [10.1109/CVPR.2009.5206815](https://doi.org/10.1109/CVPR.2009.5206815).

7. Atikur Rahman and Sachin Patel. *Annotated Potholes Image Dataset*. Kaggle, 2020. DOI: [10.34740/KAGGLE/DSV/973710](https://www.kaggle.com/dsv/973710).
 


