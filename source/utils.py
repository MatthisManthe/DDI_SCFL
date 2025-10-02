import os
import numpy as np
import glob
import pandas as pd
import torchvision
import random
import json
import matplotlib.pyplot as plt
import shutil


# generate_train_val_GTA("/home/manthe/Documents/PhD_works/advanced_sample_level_cffl/datasets/GTA5", "/home/manthe/Documents/PhD_works/advanced_sample_level_cffl/datasets/GTA5_train_val")
def generate_train_val_test_GTA(data_path, target_folder, nb_train=2518, nb_val=457, nb_test=500, seed=0):
    
    im_dir = os.path.join(data_path, "images")
    label_dir = os.path.join(data_path, "labels")
    
    im_paths = glob.glob(os.path.join(im_dir, "*"))
    filenames = np.array([os.path.basename(file) for file in im_paths])
        
    np.random.seed(seed)
    perm = np.random.permutation(len(im_paths))
    
    train_set = filenames[perm[:nb_train]]
    val_set = filenames[perm[nb_train:nb_train+nb_val]]
    test_set = filenames[perm[nb_train+nb_val:nb_train+nb_val+nb_test]]
    
    print("Training set size: ", len(train_set), "Validation set size: ", len(val_set))
    
    for file in train_set:
        shutil.copy(os.path.join(im_dir, file), os.path.join(target_folder, "images", "train", file))
        shutil.copy(os.path.join(label_dir, file), os.path.join(target_folder, "labels", "train", file))
        
    for file in val_set:
        shutil.copy(os.path.join(im_dir, file), os.path.join(target_folder, "images", "val", file))
        shutil.copy(os.path.join(label_dir, file), os.path.join(target_folder, "labels", "val", file))
        
    for file in test_set:
        shutil.copy(os.path.join(im_dir, file), os.path.join(target_folder, "images", "test", file))
        shutil.copy(os.path.join(label_dir, file), os.path.join(target_folder, "labels", "test", file))
    
    return 0
