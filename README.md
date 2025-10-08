# Deep Domain Isolation and Sample Clustered Federated Learning for Semantic Segmentation

This is the official code base for the paper 

> [Matthis Manthe et al., “Deep Domain Isolation and Sample Clustered Federated Learning for Semantic Segmentation,” in Machine Learning and Knowledge Discovery in Databases. Research Track, ed. Albert Bifet et al. (Springer Nature Switzerland, 2024)](https://doi.org/10.1007/978-3-031-70359-1_22)
## Code base structure
The structure of the code base is simple
- Directly in the ```/source/model_training``` folder are different python codes to train a segmentation model with the variety of algorithms, datasets and data distributions described in the paper (*Centralized*, *FedAvg*, *SCAFFOLD*, *Local finetuning*, *CFL*, and *SCFL*) as well as the code to train a domain classifier in a federated fashion,
- In the folder ```/source/model_test``` are the testing codes producing final performances based on the output of training algorithms,
- In the ```/source``` folder directly can be found the two python codes used to compute a federated clustering of images using *DDI* as described in the paper (one for each dataset).
- In the directory ```/config``` can be found one example json config file for each executable python code.

To reproduce the experiments obtained with SCFL in our paper GTA5+Cityscapes, one would typically have to train a model with *FedAvg* using ```/source/model_training/Fed_GTA5_Cityscapes.py```, compute a federated clustering based on the obtained model using ```/source/Spectral_GMM_GTA_Cityscapes.py```, finetune a federated model per cluster using ```/source/model_training/Finetune_Fed_cluster_GTA5_Cityscapes.py``` and test them using ```/source/model_test/Test_Finetune_Fed_cluster_GTA5_Cityscapes.py``` (the crunch was real ...).

## Launching an experiment
All these python files can be ran using the following typical command

```
python3 NAME.py --config_path CONFIG_NAME.json
```

which, for FedAvg on the combination of GTA5+Cityscapes, becomes 

```
python3 source/model_training/Fed_GTA5_Cityscapes.py --config_path config/config_Fed_GTA5_Cityscapes.json
```

One needs to create a ```/runs``` directory for experiment folders to be created every time a code is ran, containing everything related to the experiment instance (tensorboard, model weights, copy of the config file, figures, etc.).
 
## Datasets
The Triple MNIST Segmentation 0134 dataset used in the experiment can be found on zenodo: https://doi.org/10.5281/zenodo.17276525.

To reproduce the experiments on the combination of Cityscapes and GTA5 datasets.
- Based on the Cityscapes dataset from "Marius Cordts et al., “The Cityscapes Dataset for Semantic Urban Scene Understanding,” 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), IEEE, June 2016, 3213–23, https://doi.org/10.1109/CVPR.2016.350.", one must structure the downloaded folder as follows 
```
Cityscapes
├── gtFine
│   ├── train
│   ├── val
│   └── test
└── leftImg8bit
	 ├── train
 	 ├── val
	 └── test
```

with the following selected cities in each of the ```train```, ```val``` and ```test``` directories, with images and segmentations maps in the ```leftImg8bit``` and ```gtFine``` directories respectively
```
train
├── aachen
├── bochum
├── bremen
├── cologne
├── darmstadt
├── erfurt
├── hamburg
├── hanover
├── jena
├── krefeld
├── strasbourg
├── stuttgart
├── tubingen
├── ulm
└── zurich
```

```
val
├── dusseldorf
├── monchengladbach
└── weimar
```

```
test
├── frankfurt
├── lindau
└── munster
```

- Based on the GTA5 dataset from "Stephan R. Richter et al., “Playing for Data: Ground Truth from Computer Games,” 2016 European Conference on Computer Vision (ECCV), arXiv:1608.02192", one must structure the downloaded folder as follows

```
GTA5
├── images
│   ├── train
│   ├── val
│   └── test
└── labels
	 ├── train
 	 ├── val
	 └── test
```

As we used a subset of the images of the original dataset, the list of randomly selected images for each set can be found in the folder ```/Cityscapes_GTA5_data/GTA5_selected_samples```.

The generated partitionings (*IID*, *Full non IID* and *Dirichlet non IID*) for Cityscapes+GTA5 can be found in the folder ```Cityscapes_GTA5_data/Cityscapes_GTA5_partitionings```.

## Notable python codes
- ```/source/data/generator_multiple_mnist.py``` contains the code to generate the Triple MNIST segmentation dataset. **The gray-scale inversions described in the paper were performed when loading the dataset in the training codes of the experiments, not at the creation of the dataset.**
- ```/source/data/partitioning_utils.py``` contains the code to generate the IID, Full non IID and Dirichlet non IID distributions from an existing dataset.



## Dependencies
The main frameworks used are essentially 
- Pytorch (2.0.1)
- Torchvision (0.15.2)
- Monai (1.3.0)
- Sklearn (1.3.2)
- Numpy

with additional dependencies with tqdm, glob, pandas, pickle, skimage and pacmap.


