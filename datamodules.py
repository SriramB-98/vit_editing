from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
import torchvision.transforms as transforms
import glob
from PIL import Image
import lightning as pl
import torch
from tools.datasets import ImageNet, ImageNet9
from create_toy_dataset import ToyDataset

class WaterbirdsDataModule(pl.LightningDataModule):
    def __init__(self, root_dir='/cmlscratch/mmoayeri/data', batch_size=64, num_workers=4):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = 2

    def setup(self, stage=None):
        waterbirds_dataset = get_dataset("waterbirds", root_dir=self.root_dir)
        # an alternate transformation can also be used -- couldnt find the exact original size
        transform = transforms.Compose([transforms.Resize(224), 
                                        transforms.CenterCrop(224), 
                                        transforms.ToTensor()
                                       ])
        self.train_dset = waterbirds_dataset.get_subset("train", transform=transform)
        self.test_dset = waterbirds_dataset.get_subset("test", transform=transform)
        
    def train_dataloader(self):
        return get_train_loader("standard", self.train_dset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return get_eval_loader("standard", self.test_dset, batch_size=4*self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return self.val_dataloader()
    

class ToyDataModule(pl.LightningDataModule):
    def __init__(self, root_dir='/cmlscratch/sriramb/toy_datasets/', batch_size=64, num_workers=4, get_masks=False):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = 2
        self.get_masks = get_masks

    def setup(self, stage=None):
        # an alternate transformation can also be used -- couldnt find the exact original size
        transform = transforms.Compose([transforms.Resize(224), 
                                        transforms.CenterCrop(224), 
                                        transforms.ToTensor() ])
        self.train_dset, self.val_dset = torch.utils.data.random_split(ToyDataset(self.root_dir+'/corr/', 
                                                                                  transform=transform,
                                                                                  get_masks=self.get_masks), 
                                                                        [0.8, 0.2])
        self.test_dset = ToyDataset(self.root_dir+'/uncorr/', 
                                    transform=transform, 
                                    get_masks=self.get_masks)
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dset, 
                                           batch_size=self.batch_size, 
                                           shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return [torch.utils.data.DataLoader(self.val_dset, 
                                            batch_size=self.batch_size,
                                            shuffle=False, num_workers=self.num_workers),
                torch.utils.data.DataLoader(self.test_dset, 
                                           batch_size=self.batch_size, 
                                           shuffle=False, num_workers=self.num_workers)]
                

    def test_dataloader(self):
        return self.val_dataloader()

    
class ImageNet9DataModule(pl.LightningDataModule):
    def __init__(self, root_dir='/cmlscratch/sriramb/imagenet9', get_masks=False, batch_size=64, num_workers=4):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = 9
        self.get_masks = get_masks

    def setup(self, stage=None):
        self.orig_dataset = ImageNet9(f'{self.root_dir}/original', masks=self.get_masks)
        self.mixedrand_dataset = ImageNet9(f'{self.root_dir}/mixed_rand', masks=self.get_masks)
        self.mixedsame_dataset = ImageNet9(f'{self.root_dir}/mixed_same', masks=self.get_masks)
        self.onlybg_dataset = ImageNet9(f'{self.root_dir}/only_bg_t', masks=self.get_masks)
        self.nofg_dataset = ImageNet9(f'{self.root_dir}/no_fg', masks=self.get_masks)
        self.onlyfg_dataset = ImageNet9(f'{self.root_dir}/only_fg', masks=self.get_masks)
#         self.fgmask_dataset = ImageNet9(f'{self.root_dir}/fg_mask', masks=True)
        
    def train_dataloader(self):
        return self.orig_dataset.make_loaders(self.num_workers, self.batch_size, 
                                          mode='train', shuffle_val=True)

    def val_dataloader(self):
        return [self.orig_dataset.make_loaders(self.num_workers, 4*self.batch_size, 
                                               mode='val', shuffle_val=False),
                self.mixedrand_dataset.make_loaders(self.num_workers, 4*self.batch_size, 
                                                   mode='val', shuffle_val=False),
                self.mixedsame_dataset.make_loaders(self.num_workers, 4*self.batch_size, 
                                                   mode='val', shuffle_val=False)]
    def test_dataloader(self):
        return self.val_dataloader()
    
