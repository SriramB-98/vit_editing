# import imagenet_models
import torch as ch
import os
from torchvision import transforms as T
from tools import folder
from torch.utils.data import DataLoader


def make_loaders(workers, batch_size, transforms, data_path, dataset, mode='val', masks=False, subset_ratio=1, shuffle_val=False, rand_split=False):
    '''
    '''
    print(f"==> Preparing dataset {dataset}..")

    test_path = os.path.join(data_path, mode)
    if not os.path.exists(test_path):
        raise ValueError("Test data must be stored in {0}".format(test_path))

    test_set = folder.ImageFolder(root=test_path, transform=transforms, masks=masks)
#     test_set = ch.utils.data.Subset(test_set, ch.randperm(int(len(test_set))).tolist()[:int(len(test_set)*subset_ratio)] )
    if rand_split:
        train_set, val_set = ch.utils.data.random_split(test_set, [int(len(test_set)*subset_ratio), len(test_set)-int(len(test_set)*subset_ratio)])
        return DataLoader(train_set, batch_size=batch_size, 
                shuffle=shuffle_val, num_workers=workers, drop_last=False, pin_memory=True),\
               DataLoader(val_set, batch_size=batch_size, 
                shuffle=False, num_workers=workers, drop_last=False, pin_memory=True)
    else:
        return DataLoader(test_set, batch_size=batch_size, 
                shuffle=shuffle_val, num_workers=workers, drop_last=False, pin_memory=True)



class DataSet(object):
    '''
    '''

    def __init__(self, ds_name, data_path, **kwargs):
        """
        """
#         required_args = ['num_classes', 'mean', 'std', 'transform_test']
#         assert set(kwargs.keys()) == set(required_args), "Missing required args, only saw %s" % kwargs.keys()
        self.ds_name = ds_name
        self.data_path = data_path
        self.__dict__.update(kwargs)

    def make_loaders(self, workers, batch_size, mode='val', transforms=None, subset_ratio=1, shuffle_val=False, rand_split=False):

        #T.Compose([self.transform_test, T.Normalize(mean=self.mean, std=self.std)])
        return make_loaders( workers=workers,
                                batch_size=batch_size,
                                transforms=self.transforms,
                                data_path=self.data_path,
                                dataset=self.ds_name,
                                mode=mode,
                                masks=self.masks,
                                subset_ratio=subset_ratio,
                                shuffle_val=shuffle_val,
                                rand_split=rand_split)
    
    def get_model(self, arch, pretrained):
        '''
        Args:
            arch (str) : name of architecture 
            pretrained (bool): whether to try to load torchvision 
                pretrained checkpoint
        Returns:
            A model with the given architecture that works for each
            dataset (e.g. with the right input/output dimensions).
        '''

        raise NotImplementedError

class ImageNet9(DataSet):
    '''
    '''
    def __init__(self, data_path, masks=False):
        """
        """
        ds_name = 'ImageNet9'
        ds_kwargs = {
            'num_classes': 9,
            'mean': ch.tensor([0.4717, 0.4499, 0.3837]), 
            'std': ch.tensor([0.2600, 0.2516, 0.2575]),
            'transforms':T.Compose([T.Resize(224), 
                                        T.CenterCrop(224), 
                                        T.ToTensor()
                                       ]),
            'masks':masks
        }
        super(ImageNet9, self).__init__(ds_name,
                data_path, **ds_kwargs)
        
#     def get_model(self, arch, pretrained):
#         """
#         """
#         if pretrained:
#             raise ValueError("Dataset doesn't support pytorch_pretrained")
#         return imagenet_models.__dict__[arch](num_classes=self.num_classes)

class ImageNet(DataSet):
    '''
    '''
    def __init__(self, data_path):
        """
        """
        ds_name = 'ImageNet'
        ds_kwargs = {
            'num_classes': 1000,
            'mean': ch.tensor([0.485, 0.456, 0.406]),
            'std': ch.tensor([0.229, 0.224, 0.225]),
            'transform_test': T.ToTensor()
        }
        super(ImageNet, self).__init__(ds_name,
                data_path, **ds_kwargs)
        
#     def get_model(self, arch, pretrained):
#         """
#         """
#         return imagenet_models.__dict__[arch](num_classes=self.num_classes, 
#                                         pretrained=pretrained)

