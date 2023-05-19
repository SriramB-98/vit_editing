import lightning as pl
from torchvision.models import vit_b_16, ViT_B_16_Weights
from madry_models import deit_base_patch16_224 as deit_b_16
from torchvision import transforms
from sal_resnet import resnet50
from torch import nn, optim, Tensor
import torch

class Predictor(pl.LightningModule):
    def __init__(self, mtype, num_classes, lr=1e-3, weight_decay=1e-5):
        super().__init__()
        self.model = globals()[mtype](pretrained=True)
        if 'vit' in mtype:
            fc_dims=(768, num_classes)
            fc_layer='heads'
        elif 'deit' in mtype:
            fc_dims=(768, num_classes)
            fc_layer='head'
        elif 'resnet' in mtype:
            fc_dims=(2048, num_classes)
            fc_layer='fc'
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()
        setattr(self.model, fc_layer, nn.Linear(*fc_dims))
        self.normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.validation_step_outputs = []
    
    def forward(self, x):
        return self.model(self.normalizer(x))
      
    def get_accuracy(self, preds, y):
        hits = (preds.argmax(dim=1) == y).float()
        return hits.mean()
        
    def configure_optimizers(self):
        self.trainer.fit_loop.setup_data()
        self.steps_per_epoch = len(self.trainer.train_dataloader) // self.trainer.accumulate_grad_batches
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(epoch=float(self.global_step)/self.steps_per_epoch) 
        


class WaterbirdsPredictor(Predictor):  
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y, _ = batch
        preds = self.forward(x)
        loss = nn.functional.cross_entropy(preds, y)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        self.log("train_acc", self.get_accuracy(preds, y))
        return loss
    
    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y, z = batch
        preds = self.forward(x)
        self.log("test_acc", self.get_accuracy(preds, y))
        self.validation_step_outputs.append((preds.argmax(dim=1), y, z))
        
    def validation_step(self, batch, batch_idx):
        x, y, z = batch
        preds = self.forward(x)
        self.log("val_acc", self.get_accuracy(preds, y))
        self.validation_step_outputs.append((preds.argmax(dim=1), y, z))
#         return preds.argmax(dim=1)
        
    def epoch_level_metrics(self, mode):
        all_preds, all_y, all_z = zip(*self.validation_step_outputs)
        hits = torch.cat(list(all_preds)) == torch.cat(list(all_y))
        all_z = torch.cat(all_z).bool()
        self.log(f"{mode}_acc_00", hits[ (~all_z[:,0]) & (~all_z[:,1])].float().mean())
        self.log(f"{mode}_acc_01", hits[(~all_z[:,0]) & all_z[:,1]].float().mean())
        self.log(f"{mode}_acc_10", hits[all_z[:,0] & (~all_z[:,1])].float().mean())
        self.log(f"{mode}_acc_11", hits[all_z[:,0] & all_z[:,1]].float().mean())
        self.validation_step_outputs.clear()
        
    def on_validation_epoch_end(self):
        self.epoch_level_metrics("val")
        
    def on_test_epoch_end(self):
        self.epoch_level_metrics("test")


class ImageNet9Predictor(Predictor):  
    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = nn.functional.cross_entropy(preds, y)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        self.log("train_acc", self.get_accuracy(preds, y))
        return loss
    
    def test_step(self, batch, batch_idx, dataloader_idx):
        # this is the test loop
        names = ('orig', 'mixed_rand', 'mixed_same')
        x, y = batch
        preds = self.forward(x)
        self.log(f"{names[dataloader_idx]}_test_acc", self.get_accuracy(preds, y), add_dataloader_idx=False)
        
    def validation_step(self, batch, batch_idx, dataloader_idx):
        names = ('orig', 'mixed_rand', 'mixed_same')
        x, y = batch
        preds = self.forward(x)
        self.log(f"{names[dataloader_idx]}_val_acc", self.get_accuracy(preds, y), add_dataloader_idx=False)

class ToyDatasetPredictor(Predictor):  
    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = nn.functional.cross_entropy(preds, y)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        self.log("train_acc", self.get_accuracy(preds, y))
        return loss
    
    def test_step(self, batch, batch_idx, dataloader_idx):
        # this is the test loop
        names = ('corr', 'uncorr')
        x, y = batch
        preds = self.forward(x)
        self.log(f"{names[dataloader_idx]}_test_acc", self.get_accuracy(preds, y), add_dataloader_idx=False)
        
    def validation_step(self, batch, batch_idx, dataloader_idx):
        names = ('corr', 'uncorr')
        x, y = batch
        preds = self.forward(x)
        self.log(f"{names[dataloader_idx]}_val_acc", self.get_accuracy(preds, y), add_dataloader_idx=False)
