# Deep Domain Isolation and Sample Clustered Federated Learning for Semantic Segmentation

This is the official code base for the paper 

> [Matthis Manthe et al., “Deep Domain Isolation and Sample Clustered Federated Learning for Semantic Segmentation,” in Machine Learning and Knowledge Discovery in Databases. Research Track, ed. Albert Bifet et al. (Springer Nature Switzerland, 2024)](https://doi.org/10.1007/978-3-031-70359-1_22)


## Notable python codes
- ```/source/data/generator_multiple_mnist.py``` contains the code to generate the Triple MNIST segmentation dataset. While the gray-scale inversions were performed when loading the dataset in the training codes for these experiments, we also added lighter types of noise we explored a bit (speckle, gaussian noise, canny edges, etc.) to build additional feature domains, but did not make it into the publication.
- ```/source/data/partitioning_utils.py``` contains the code to generate the IID, Full non IID and Dirichlet non IID distributions from an existing dataset.
 
## Datasets
- The actual datasets used in these experiments can be found on zenovo:
1. Triple MNIST seg,
2. GTA+Cityscapes.


## Code base structure
The structure of the code base is simple
- Directly in the ```/source``` folder is one python code for each training algorithm tested in the paper: *Centralized*, *Local training on institution 1*, *FedAvg* (with fixed local epochs and local iterations), *FedAvg IID*, *FedAdam*, *FedNova*, *SCAFFOLD*, *q-FedAvg*, *FedPIDAvg*, *Local finetuning*, *Ditto*, *FedPer* (accounting for *FedPer* and *LG-FedAvg*), *CFL* and *prior clustered FedAvg*,
- In the directory ```source/config``` can be found one json config file for each training algorithm necessary to start an experiment. They are copies of each config file selected through grid searches, used to generate the final results of our paper (for one fold). Note that the hyperparameters for personalized methods were selected per institution, requiring the full grid search to replicate the results of the paper.
- In the folder ```/source/test``` are the testing codes producing final performances based on the output of training algorithms (with associated ```/source/test/config``` folder with json files).

The fold splits of FeTS2022 dataset used in cross-validations can be found in the folder ```/data_folds```.

## Launching an experiment
All these python files can be ran using the following typical command

```python3 NAME.py --config_path config/CONFIG_NAME.json```

which, for FedAvg using the config file for the FeTS2022 challenge partitioning defined in ```config_FedAvg.json```, becomes 

```python3 Train_FedAvg_3D_multi_gpu.py --config_path config/config_FedAvg.json```

One needs to create a ```/runs``` directory for experiment folders to be created every time a code is ran, containing everything related to the experiment instance (tensorboard, model weights, copy of the config file, figures, etc.).

## Dependencies
The main frameworks used are essentially 
- Pytorch
- Torchvision
- Numpy
- Sklearn
- Monai

with additional dependencies with tqdm, glob, pandas, pickle, skimage and pacmap.


