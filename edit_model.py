import argparse
from datamodules import ImageNet9DataModule
import torch
from utils import *
from inspect_utils import *
from imagenet9 import get_im9_val_acc
from lightning_modules import *
from torch.nn.utils.parametrizations import orthogonal
from functools import partial

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_params(save_path, weights_dir, bases_dim=768):
    weights_bases_param = torch.load(save_path+f'/{weights_dir}/weights_bases_param.pt')
    weights_param, bases_modules_sd = weights_bases_param['weights'], weights_bases_param['bases']
    bases_modules = dict([(hk, orthogonal(torch.nn.Linear(bases_dim, bases_dim, bias=False))) for hk in bases_modules_sd.keys()])
    for k, v in bases_modules.items():
        v.load_state_dict(bases_modules_sd[k])
    return weights_param, bases_modules

# def modify_acts(input, output, name, rel_bases, ):
#     return output - output@rel_bases.T@rel_bases

def add_hooks(model, hook_names, orig_acts_dict, weights_param, bases_modules, thresh=0.05):
    for hook_name in hook_names:
        weights = (weights_param[hook_name] > thresh).float().to(device)
        bases = bases_modules[hook_name].to(device)
        modify_fn = partial(modify_output, orig_acts_dict=orig_acts_dict, 
                                           weights_param=weights, 
                                           bases_module=bases)
        my_getattr(model, hook_name).register_forward_hook(get_hook(hook_name, modify_fn=modify_fn))
    return 
    
# def edit_model(model, hook_to_modules, weights_param, bases_modules, thresh=0.05): 
#     for name, module in model.named_modules():
#         if isinstance(module, torch.nn.Linear):
#             if name in weights_param.keys():
#                 weights, bases = weights_param[name], bases_modules[name].weight
#                 mask = weights > thresh
#                 rel_bases = bases.weight.data[mask]
#                 module.weight.data = module.weight.data@(torch.eye(rel_bases.shape[1]) - rel_bases.T@rel_bases) 
#                 print(f"Edit {name} with {weights.shape} and {bases.shape}")
#     return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='./causal_neurons')
    parser.add_argument('--weights_dir', type=str, default=None)
    parser.add_argument('--bases_dim', type=int, default=768)
    parser.add_argument('--model', type=str, default='deit_b_16')
    parser.add_argument('--dataset', type=str, default='imagenet9')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--gpu_size', type=int, default=512)
    parser.add_argument('--edit', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed) 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model == 'deit_b_16' and args.dataset == 'imagenet9':
        batch_size = max(args.gpu_size*torch.cuda.device_count(), 100) if args.batch_size is None else args.batch_size
        ckpt_path = './models/imagenet9/mtype-deit_b_16_num_classes-9_lr-1e-05_weight_decay-1e-06/epoch=01-step=1200-orig_val_acc=0.99.ckpt'
        lightning_model = ImageNet9Predictor.load_from_checkpoint(ckpt_path)
        model = lightning_model.model.to(device)
        normalizer = lightning_model.normalizer
        try:
            orig_acts_dict = torch.load(ckpt_path.replace('.ckpt', '_orig_val_acts.ckpt'))
        except FileNotFoundError:
            data_module = ImageNet9DataModule(get_masks=True)
            data_module.setup()
            dataloader = data_module.orig_dataset.make_loaders(args.num_workers, batch_size, mode='val', shuffle_val=False)
            orig_acts_dict = get_intermediate_acts(model, normalizer, dataloader, ['blocks.8', 'blocks.9', 'blocks.10', 'norm'])
            torch.save(orig_acts_dict, ckpt_path.replace('.ckpt', '_orig_val_acts.ckpt'))
    else:
        raise NotImplementedError
    

    if args.edit:
        weights_param, bases_modules = load_params(args.save_path, args.weights_dir, args.bases_dim)
        hook_names = list(weights_param.keys())
        orig_acts_dict = dict([(hn, orig_acts_dict[hn].to(device)) for hn in hook_names])
        add_hooks(model, hook_names, orig_acts_dict, weights_param, bases_modules)

    orig_acc = get_im9_val_acc(model, normalizer, num_workers=args.num_workers, batch_size=batch_size, mode='original')
    mr_acc = get_im9_val_acc(model, normalizer, num_workers=args.num_workers, batch_size=batch_size, mode='mixed_rand')
    ms_acc = get_im9_val_acc(model, normalizer, num_workers=args.num_workers, batch_size=batch_size, mode='mixed_same')

    print('Original val acc: ', orig_acc)
    print('Mixed rand val acc: ', mr_acc)
    print('Mixed same val acc: ', ms_acc)