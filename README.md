# Scene Classification with MobileNet

## Project Overview
This project focuses on training a CNN (Convolutional Neural Network) image classifier capable of distinguishing three types of natural scenery: coast, forest, and mountain. Employing transfer learning, the classifier is built on MobileNet [1], known for its efficient trade-off between latency and accuracy. The goal is to develop a lightweight yet performant model suitable for mobile and embedded applications.

Scene classification poses a challenge in computer vision, requiring the classifier not only to recognize objects but also understand their layout and contextual content. Despite these challenges, it remains an active area of research due to its applications in image retrieval, robotics, self-driving cars, and disaster detection.

## Dataset
The dataset comprises images from the SUN 09 Dataset [2], focusing on three classes: coast, forest, and mountain. The dataset, available for download as [sun09.tar](http://groups.csail.mit.edu/vision/Hcontext/data/sun09.tar) (5.2GB), was designed to leverage contextual information in images. We extracted a total of 1158 images, organized into three subfolders based on classification. The prepared data, available as `data_sun09.zip` in the [data_prep_eda](data_prep_eda/) folder, is structured for training/validation/testing. The data preparation code is in the `Data_Preparation.ipynb` notebook in the same folder.

## EDA
Explore the analysis of the image dataset in the `EDA.ipynb` notebook in the [data_prep_eda](data_prep_eda/) folder.

## Model Training
Initially, we trained the Xception model, achieving excellent results, before experimenting with MobileNet for comparison. Generally, MobileNet models performed slightly worse. We fine-tuned the learning rate and applied data augmentation (vertical flip, horizontal and vertical translation, zoom-in/out). The best model, with data augmentation and a learning rate of 0.01, achieved 100% validation accuracy by epoch 3.

Find model training notebooks and data resources in the [train_model](train_model/) folder. Model training was executed on cloud.

### Exporting training notebook to script
The logic for model training is exported to a separate Python script, `model_training.py`, in the [train_model](train_model/) folder.

## Dependency and Environment Management
This project runs in a Conda environment, with the conda env file as `environment.yml` in the root folder. Set up the local environment:

1. Open a terminal in the working directory.
2. To create the conda environment from the file:
    ```
    conda env create -f environment.yml
    ```
3. To activate the environment:
    ```
    conda activate sc_env
    ```

To set up the environment for running model training notebooks in Saturn Cloud with Tensorflow and GPU, follow the provided steps.

## Containerization
The saved model (in h5 format) is converted to a TF-Lite version and deployed to AWS Lambda in a Docker container. Instructions for building and running the container are available in [containerize.md](containerize_deploy/containerize.md) in the [containerize_deploy](containerize_deploy/) folder. Additional containerization resources are provided in the same folder.

## Cloud Deployment
Details of deploying the service as an AWS Lambda function and testing it are outlined in [deploy_aws_lambda.md](containerize_deploy/deploy_aws_lambda.md) in the [containerize_deploy](containerize_deploy/) folder.
