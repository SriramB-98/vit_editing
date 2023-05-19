import numpy as np
from PIL import Image
import os
import torch
import torchvision
import shutil


def add_circle(img_array, radius, pos):
    x, y = pos
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            if (i - x) ** 2 + (j - y) ** 2 <= radius ** 2:
                img_array[i, j,:] = 255
    return

def add_square(img_array, radius, pos):
    x, y = pos
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            if (abs(i - x) <= radius) and (abs(j - y) <= radius):
                img_array[i, j,:] = 255
    return

def add_triangle(img_array, radius, pos):
    x, y = pos
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            if (abs(i - x) + abs(j - y) <= radius) and (i > x) :
                img_array[i, j,:] = 255
    return

def add_semicircle(img_array, radius, pos):
    x, y = pos
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            if (i - x) ** 2 + (j - y) ** 2 <= radius ** 2 and j >= y:
                img_array[i, j,:] = 255
    return 

def create_dataset(num_samples, corr=0.5, std=20):
    img_size = 224
    radius = 50
    num_classes = 2
    num_samples_per_class = num_samples // num_classes    
    X_list = []
    z_list = []
    for i in range(num_classes):
        X = np.zeros((num_samples_per_class, img_size, img_size, 3), dtype=np.uint8)
        z = []
        for j in range(num_samples_per_class):
            rf = np.random.rand()
            gr = np.random.randn()
            gr2 = np.random.randn()
            if i == 0:
                add_circle(X[j], radius, (img_size // 2 +gr2*std, img_size // 2 + gr*std))
                if rf < corr:
                    # make background red
                    X[j, :, :, 0] = 255

                    # add_semicircle(X[j], radius, (3*img_size // 4, img_size // 2 + gr2*std))
                    z.append(0)
                else:
                    # make background blue
                    X[j, :, :, 2] = 255

                    # add_triangle(X[j], radius, (3*img_size // 4, img_size // 2 + gr2*std))
                    z.append(1)
            elif i == 1:
                add_square(X[j], radius, (img_size // 2 +gr2*std, img_size // 2 + gr*std))
                if rf < corr:
                    X[j, :, :, 2] = 255
                    # add_triangle(X[j], radius, (3*img_size // 4, img_size // 2 + gr2*std))
                    z.append(1)
                else:
                    X[j, :, :, 0] = 255
                    # add_semicircle(X[j], radius, (3*img_size // 4, img_size // 2 + gr2*std))
                    z.append(0)
        X_list.append(X)
        z_list.append(z)
    return X_list, z_list


def create_dataset_2(num_samples, corr=0.5, std=20):
    img_size = 224
    radius = 50
    num_classes = 2
    num_samples_per_class = num_samples // num_classes    
    X_list = []
    z_list = []
    for i in range(num_classes):
        X = np.zeros((num_samples_per_class, img_size, img_size, 3), dtype=np.uint8)
        z = []
        for j in range(num_samples_per_class):
            rf = np.random.rand()
            gr = np.random.randn()
            gr2 = np.random.randn()
            if i == 0:
                X[j, :, :, 0] = 255
                if rf < corr:
                    # make background red
                    add_circle(X[j], radius, (img_size // 2 +gr2*std, img_size // 2 + gr*std))
                    # add_semicircle(X[j], radius, (3*img_size // 4, img_size // 2 + gr2*std))
                    z.append(0)
                else:
                    # make background blue
                    
                    add_square(X[j], radius, (img_size // 2 +gr2*std, img_size // 2 + gr*std))
                    # add_triangle(X[j], radius, (3*img_size // 4, img_size // 2 + gr2*std))
                    z.append(1)
            elif i == 1:
                X[j, :, :, 2] = 255
                if rf < corr:
                    add_square(X[j], radius, (img_size // 2 +gr2*std, img_size // 2 + gr*std))
                    # add_triangle(X[j], radius, (3*img_size // 4, img_size // 2 + gr2*std))
                    z.append(1)
                else:
                    add_circle(X[j], radius, (img_size // 2 +gr2*std, img_size // 2 + gr*std))
                    # add_semicircle(X[j], radius, (3*img_size // 4, img_size // 2 + gr2*std))
                    z.append(0)
        X_list.append(X)
        z_list.append(z)
    return X_list, z_list


class ToyDataset(torchvision.datasets.VisionDataset):
    def __init__(self, root, transform=None, target_transform=None, get_masks=False):
        super(ToyDataset, self).__init__(root, transform=transform, target_transform=target_transform)
        self.img_paths_0 = os.listdir(root+'/0')
        self.img_paths_1 = os.listdir(root+'/1')
        self.img_paths = self.img_paths_0 + self.img_paths_1
        self.labels = [0]*len(self.img_paths_0) + [1]*len(self.img_paths_1)
        self.get_masks = get_masks
        return

    def __getitem__(self, index):
        target = self.labels[index]
        img = Image.open(os.path.join(self.root, str(target), self.img_paths[index]))
        # conf = int(self.img_paths[index].split('_')[1].split('.')[0])
        if self.transform is not None:
            img = self.transform(img)
            img = img.expand(3, -1, -1)
        if self.target_transform is not None:
            target = self.target_transform(target) 
        if self.get_masks:
            mask = (img.mean(dim=0, keepdim=True) == 1.).float()
            return img, mask, target
        else:
            return img, target

    def __len__(self):
        return len(self.img_paths)

if __name__ == '__main__':
    
    # save dataset in image folder
    dset_names = ['corr', 'uncorr']
    main_dname = '/cmlscratch/sriramb/toy_datasets'
    create_fn = create_dataset
    for dset_name in dset_names:
        X_list, z_list = create_fn(1000, 1.0 if dset_name == 'corr' else 0.5)
        for i, (Xi, zi) in enumerate(zip(X_list, z_list)):
            shutil.rmtree(f'{main_dname}/{dset_name}/{i}', ignore_errors=True)
            os.makedirs(f'{main_dname}/{dset_name}/{i}', exist_ok=True)
            for j, (Xij, zij) in enumerate(zip(Xi, zi)):
                img = Image.fromarray(Xij)
                img.save(f'{main_dname}/{dset_name}/{i}/{j}_{zij}.png')
