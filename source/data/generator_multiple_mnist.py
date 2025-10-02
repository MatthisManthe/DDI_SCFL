import sys
import os
import os.path as osp
import numpy as np
import argparse
from torchvision import datasets
import torchvision.transforms as transforms
from matplotlib import colors
import matplotlib.pyplot as plt
import torch
import json
from tqdm import  tqdm
from PIL import Image
from skimage.filters import gaussian
from skimage import feature


def sample_coordinate(high, size):
    if high > 0:
        return np.random.randint(high, size=size)
    else:
        return np.zeros(size).astype(np.int)
    

def generate_triple_set(dataset, indexes, classes, nb_image_per_class, h, w, config, name):
    
    # -------------------- Generate samples with MNIST data -----------------
    if not os.path.exists(os.path.join(config["datasets_path"], config["multimnist_path"], name)):
        os.makedirs(os.path.join(config["datasets_path"], config["multimnist_path"], name))
        
    if config["save_type"] == "numpy":
        all_images = []
        all_masks = []
        all_labels = []
    
    for current_class in tqdm(classes):
        class_str = str(current_class)
        class_str = class_str = '0'*(config["num_digit"]-len(class_str))+class_str
        digits = [int(c) for c in str(class_str)]
        print("\nDigits: ", digits)
        
        for k in range(nb_image_per_class):
            imgs = [np.squeeze(dataset.data[np.random.choice(indexes[d])]) for d in digits]
            masks = [(imgs[i]>0)*(1+digits[i]) for i in range(len(imgs))]
            
            background = np.zeros((config["image_size"])).astype(np.uint8)
            mask_background = np.zeros((config["image_size"])).astype(np.uint8)
            
            # sample coordinates
            ys = sample_coordinate(config["image_size"][0]-h, config["num_digit"])
            xs = sample_coordinate(config["image_size"][1]//config["num_digit"]-w,
                                   size=config["num_digit"])
            xs = [l*config["image_size"][1]//config["num_digit"]+xs[l]
                  for l in range(config["num_digit"])]
            
            # combine images
            for i in range(config["num_digit"]):
                background[ys[i]:ys[i]+h, xs[i]:xs[i]+w] = imgs[i]
                mask_background[ys[i]:ys[i]+h, xs[i]:xs[i]+w] = masks[i]
                
            if config["save_type"] == "image":
                image = Image.fromarray(background)
                mask = Image.fromarray(mask_background)
                image.save(os.path.join(config["datasets_path"], config["multimnist_path"], name, class_str+"_"+str(k)+".png"))
                mask.save(os.path.join(config["datasets_path"], config["multimnist_path"], name, class_str+"_"+str(k)+"_mask.png")) 
            
            elif config["save_type"] == "numpy":
                all_images.append(background.copy())
                all_masks.append(mask_background.copy())
                all_labels.append(class_str)
                
    if config["save_type"] == "numpy":
        all_images = np.array(all_images)
        all_masks = np.array(all_masks)
        all_labels = np.array(all_labels)
        
        if config["shuffle"]:
            permut = np.random.permutation(len(all_images))
            all_images = all_images[permut]
            all_masks = all_masks[permut]
            all_labels = all_labels[permut]
            
        print(f"\nSaving {name} images, images shape : {all_images.shape}, masks shape : {all_masks.shape}, labels shape : {all_labels.shape}")
        np.save(os.path.join(config["datasets_path"], config["multimnist_path"], name, "images.npy"), all_images)
        np.save(os.path.join(config["datasets_path"], config["multimnist_path"], name, "masks.npy"), all_masks)
        np.save(os.path.join(config["datasets_path"], config["multimnist_path"], name, "labels.npy"), all_labels)
    
  
def generate_triple_set_digit_subset(dataset, indexes, classes, class_map, nb_image_per_class, h, w, config, name):
    
    # -------------------- Generate samples with MNIST data -----------------
    if not os.path.exists(os.path.join(config["datasets_path"], config["multimnist_path"], name)):
        os.makedirs(os.path.join(config["datasets_path"], config["multimnist_path"], name))
        
    if config["save_type"] == "numpy":
        all_images = []
        all_masks = []
        all_labels = []
    
    for current_class in tqdm(classes):
        class_str = str(current_class)
        class_str = class_str = str(config["selected_digits"][0])*(config["num_digit"]-len(class_str))+class_str
        id_digits = [int(c) for c in str(class_str)]
        print("\nDigits: ", id_digits)
        
        for k in range(nb_image_per_class):
            imgs = [np.squeeze(dataset.data[np.random.choice(indexes[d])]) for d in id_digits]
            masks = [(imgs[i]>0)*(1+id_digits[i]) for i in range(len(imgs))]
            
            background = np.zeros((config["image_size"])).astype(np.uint8)
            mask_background = np.zeros((config["image_size"])).astype(np.uint8)
            
            # sample coordinates
            ys = sample_coordinate(config["image_size"][0]-h, config["num_digit"])
            xs = sample_coordinate(config["image_size"][1]//config["num_digit"]-w,
                                   size=config["num_digit"])
            xs = [l*config["image_size"][1]//config["num_digit"]+xs[l]
                  for l in range(config["num_digit"])]
            
            # combine images
            for i in range(config["num_digit"]):
                background[ys[i]:ys[i]+h, xs[i]:xs[i]+w] = imgs[i]
                mask_background[ys[i]:ys[i]+h, xs[i]:xs[i]+w] = masks[i]
                
            if config["save_type"] == "image":
                image = Image.fromarray(background)
                mask = Image.fromarray(mask_background)
                image.save(os.path.join(config["datasets_path"], config["multimnist_path"], name, class_str+"_"+str(k)+".png"))
                mask.save(os.path.join(config["datasets_path"], config["multimnist_path"], name, class_str+"_"+str(k)+"_mask.png")) 
            
            elif config["save_type"] == "numpy":
                all_images.append(background.copy())
                all_masks.append(mask_background.copy())
                all_labels.append(class_str)
             
    frequency_unique_class = np.unique(all_labels, return_counts=True)
    print(frequency_unique_class)
    
    if config["save_type"] == "numpy":
        all_images = np.array(all_images)
        all_masks = np.array(all_masks)
        all_labels = np.array(all_labels)
        
        if config["shuffle"]:
            permut = np.random.permutation(len(all_images))
            all_images = all_images[permut]
            all_masks = all_masks[permut]
            all_labels = all_labels[permut]
            
        print(f"\nSaving {name} images, images shape : {all_images.shape}, masks shape : {all_masks.shape}, labels shape : {all_labels.shape}")
        np.save(os.path.join(config["datasets_path"], config["multimnist_path"], name, "images.npy"), all_images)
        np.save(os.path.join(config["datasets_path"], config["multimnist_path"], name, "masks.npy"), all_masks)
        np.save(os.path.join(config["datasets_path"], config["multimnist_path"], name, "labels.npy"), all_labels)      
        

def plots_some_images(config, name, number):
    
    images = np.load(os.path.join(config["datasets_path"], config["multimnist_path"], name, "images.npy"))
    masks = np.load(os.path.join(config["datasets_path"], config["multimnist_path"], name, "masks.npy"))
    labels = np.load(os.path.join(config["datasets_path"], config["multimnist_path"], name, "labels.npy"))
    
    for _ in range(number):
        
        index = np.random.choice(np.arange(len(images)))
        fig, axs = plt.subplots(2,1, figsize=(10,10))
        axs[0].imshow(images[index], cmap="gray")
        
        cmap = plt.cm.hsv
        norm = colors.BoundaryNorm(np.arange(-0.5, 12.5, 1), cmap.N)
        im = axs[1].imshow(masks[index], cmap=cmap, norm=norm, alpha=0.3)
        plt.colorbar(im)
        plt.title(labels[index])
        plt.show()
    
    

def glass_blur(x, severity=1):
    # sigma, max_delta, iterations
    c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]

    x = np.uint8(gaussian(np.array(x) / 255., sigma=c[0]) * 255)

    # locally shuffle pixels
    for i in range(c[2]):
        for h in range(64 - c[1], c[1], -1):
            for w in range(96 - c[1], c[1], -1):
                if np.random.choice([True, False], 1)[0]:
                    dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                    h_prime, w_prime = h + dy, w + dx
                    # swap
                    x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]

    x = np.clip(gaussian(x / 255., sigma=c[0]), 0, 1) * 255
    return x.astype(np.float32)



def spatter(x, severity=4):
    c = [(0.65, 0.3, 4, 0.69, 0.6, 0),
         (0.65, 0.3, 3, 0.68, 0.6, 0),
         (0.65, 0.3, 2, 0.68, 0.5, 0),
         (0.65, 0.3, 1, 0.65, 1.5, 1),
         (0.67, 0.4, 1, 0.65, 1.5, 1)][severity - 1]

    x = np.array(x, dtype=np.float32) / 255.

    liquid_layer = np.random.normal(size=x.shape, loc=c[0], scale=c[1])

    liquid_layer = gaussian(liquid_layer, sigma=c[2])
    liquid_layer[liquid_layer < c[3]] = 0
    
    m = np.where(liquid_layer > c[3], 1, 0)
    m = gaussian(m.astype(np.float32), sigma=c[4])
    m[m < 0.8] = 0

    # mud spatter
    color = 63 / 255. * np.ones_like(x) * m
    x *= (1 - m)

    return np.clip(x + color, 0, 1) * 255


def canny_edges(x):
    x = np.array(x) / 255.
    x = feature.canny(x).astype(np.float32)
    return x * 255

def speckle_noise(x, severity=5):
    c = [.15, .2, 0.35, 0.45, 0.6][severity - 1]

    x = np.array(x) / 255.
    x = np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255
    return x.astype(np.float32)

def main(args, config):

    # Create a Triple MNIST dataset
    if config["generate_dataset"]:
        train_data = datasets.MNIST(
            root = config["datasets_path"],
            train = True,                         
            transform = transforms.ToTensor(), 
            download = True,
        )
        test_data = datasets.MNIST(
            root = config["datasets_path"], 
            train = False, 
            transform = transforms.ToTensor(),
            download = True
        )
        
        print(train_data.data.shape, train_data.targets.shape, test_data.data.shape, test_data.targets.shape)
        print(train_data.data[0], train_data.targets[0])
        
        plt.imshow(train_data.data[0], cmap="gray")
        plt.imshow(train_data.data[0]>0, alpha=0.2)
        
        h, w = train_data.data.shape[1:3]
    
        if len(config["selected_digits"]) == 0:
            # split: train, val, test
            num_original_class = len(np.unique(train_data.targets))
            num_class = len(np.unique(train_data.targets))**config["num_digit"]
            classes = list(np.array(range(num_class)))
            print(classes)
            
            # label index
            indexes = []
            test_indexes = []
            for c in range(num_original_class):
                indexes.append(list(np.where(train_data.targets == c)[0]))
                test_indexes.append(list(np.where(test_data.targets == c)[0]))
        
            # generate images for every class
            assert config["image_size"][1]//config["num_digit"] >= w
            np.random.seed(config["seed"])
        
            if not os.path.exists(os.path.join(config["datasets_path"], config["multimnist_path"])):
                os.makedirs(os.path.join(config["datasets_path"], config["multimnist_path"]))
        
            generate_triple_set(train_data, indexes, classes, config["num_image_per_class_train"], h, w, config, "train")
            generate_triple_set(train_data, indexes, classes, config["num_image_per_class_val"], h, w, config, "val")
            generate_triple_set(test_data, test_indexes, classes, config["num_image_per_class_test"], h, w, config, "test")
        
            if config["save_type"] == "numpy":
                plots_some_images(config, "train", 10)
                plots_some_images(config, "val", 10)
                plots_some_images(config, "test", 10)
                
        else:
            # split: train, val, test
            num_original_class = len(config["selected_digits"])
            num_class = num_original_class**config["num_digit"]
            classes = [np.base_repr(i, base=num_original_class) for i in range(num_class)]           
            class_map = {idc:config["selected_digits"][idc] for idc in range(num_original_class)}
            print(classes)
            print(class_map)
            
            # label index
            indexes = []
            test_indexes = []
            for c in config["selected_digits"]:
                indexes.append(list(np.where(train_data.targets == c)[0]))
                test_indexes.append(list(np.where(test_data.targets == c)[0]))
              
            # generate images for every class
            assert config["image_size"][1]//config["num_digit"] >= w
            np.random.seed(config["seed"])
            
            if not os.path.exists(os.path.join(config["datasets_path"], config["multimnist_path"])):
                os.makedirs(os.path.join(config["datasets_path"], config["multimnist_path"]))
                
            generate_triple_set_digit_subset(train_data, indexes, classes, class_map, config["num_image_per_class_train"], h, w, config, "train")
            generate_triple_set_digit_subset(train_data, indexes, classes, class_map, config["num_image_per_class_val"], h, w, config, "val")
            generate_triple_set_digit_subset(test_data, test_indexes, classes, class_map, config["num_image_per_class_test"], h, w, config, "test")
        
            if config["save_type"] == "numpy":
                plots_some_images(config, "train", 10)
                plots_some_images(config, "val", 10)
                plots_some_images(config, "test", 10)
            
            
    # Alter an existing Triple MNIST dataset
    elif config["alter_dataset"]:
        
        for dataset_path in config["paths_to_dataset"]:
            
            dataset = np.load(os.path.join(dataset_path, config["to_alter"]+'.npy')).astype(dtype=np.float32)
            np.random.seed(config["alter_seed"])
            
            name = ""
            if config["half_gaussian_noise"]:
                name += f"_gaussian_noise_sigma_{config['sigma']}"
            if config["half_glass_noise"]:
                name += f"_glass_noise_severity_{config['severity_glass']}"
            if config["half_spatter"]:
                name += f"_spatter_severity_{config['severity_spatter']}"
            if config["half_canny"]:
                name += "_canny"
            if config["half_speckle"]:
                name += f"_speckle_severity_{config['severity_speckle']}"
                
            if config["full_gaussian_noise"]:
                name += f"_full_gaussian_noise_sigma_{config['sigma']}"
            if config["full_glass_noise"]:
                name += f"_full_glass_noise_severity_{config['severity_glass']}"
            if config["full_spatter"]:
                name += f"_full_spatter_severity_{config['severity_spatter']}"
            if config["full_canny"]:
                name += "_full_canny"
            if config["full_speckle"]:
                name += f"_full_speckle_severity_{config['severity_speckle']}"
                
            for sample_id in tqdm(range(len(dataset)//2)):
                
                if config["full_gaussian_noise"]:
                    added_noise = np.random.normal(scale=config["sigma"], size=dataset.shape[1:]).astype(dtype=np.float32)
                    dataset[sample_id] = dataset[sample_id] + added_noise
                
                if config["full_glass_noise"]:
                    dataset[sample_id] = glass_blur(dataset[sample_id], severity=config["severity_glass"])
            
                if config["full_spatter"]:
                    dataset[sample_id] = spatter(dataset[sample_id], severity=config["severity_spatter"])
            
                if config["full_canny"]:
                    dataset[sample_id] = canny_edges(dataset[sample_id])
                    
                if config["full_speckle"]:
                    dataset[sample_id] = speckle_noise(dataset[sample_id], severity=config["severity_speckle"])
                    
                if config["show"]:
                    plt.imshow(dataset[sample_id], cmap="gray")
                    plt.show()
                
            for sample_id in tqdm(range(len(dataset)//2, len(dataset))):
                
                if config["half_gaussian_noise"] or config["full_gaussian_noise"]:
                    added_noise = np.random.normal(scale=config["sigma"], size=dataset.shape[1:]).astype(dtype=np.float32)
                    dataset[sample_id] = dataset[sample_id] + added_noise
                
                if config["half_glass_noise"] or config["full_glass_noise"]:
                    dataset[sample_id] = glass_blur(dataset[sample_id], severity=config["severity_glass"])
            
                if config["half_spatter"] or config["full_spatter"]:
                    dataset[sample_id] = spatter(dataset[sample_id], severity=config["severity_spatter"])
            
                if config["half_canny"] or config["full_canny"]:
                    dataset[sample_id] = canny_edges(dataset[sample_id])
                    
                if config["half_speckle"]:
                    dataset[sample_id] = speckle_noise(dataset[sample_id], severity=config["severity_speckle"])
                    
                if config["show"]:
                    plt.imshow(dataset[sample_id], cmap="gray")
                    plt.show()
                
            np.save(os.path.join(dataset_path, config["to_alter"]+name+'.npy'), dataset)
           



if __name__ == '__main__':
    
    print("Curent working directory: ", os.getcwd())
    
    print("Is cuda avaiable? ", torch.cuda.is_available())
    print("Number of cuda devices available: ", torch.cuda.device_count())
    
    # Define argument parser and its attributes
    parser = argparse.ArgumentParser(description='Generate triple MNIST')
    
    parser.add_argument('--config_path', dest='config_path', type=str,
                        help='path to json config file')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Read the config file
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    
    main(args, config)
