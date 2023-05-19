from utils import *
# from objectnet_test import *
import torch
torch.set_default_dtype(torch.float32)
import argparse
from datamodules import WaterbirdsDataModule, ImageNet9DataModule, ToyDataModule
from lightning_modules import *
from inspect_utils import *
from utils import dict_to_str, str_to_dict
import pickle
import torch.nn.functional as F
from torch.nn.utils.parametrizations import orthogonal
from itertools import chain
from functools import partial
import wandb
torch.multiprocessing.set_sharing_strategy('file_system')
import os
from collections import defaultdict
from imagenet9 import get_im9_val_acc
from edit_model import add_hooks
#  label_names = ['0_dog', '1_bird', '2_wheeled vehicle', '3_reptile', '4_carnivore', '5_insect', '6_musical instrument', '7_primate', '8_fish']

activation = {}

def read_output(inp, out, name):
    if isinstance(out, tuple):
        activation[name] = out[0]
    else:
        activation[name] = out
    if name == 'encoder' or name == 'norm':
        activation[name] = activation[name][:,0:1]
    return

def read_input(inp, out, name):
    if isinstance(inp, tuple):
        activation[name] = inp[0]
    else:
        activation[name] = inp
    return

def modify_output(inp, out, name, orig_acts_dict, weights_param, bases_module, misc_params):
    if orig_acts_dict[name] is None:
        return out
#     print('b')
    orig_acts = orig_acts_dict[name]
    if isinstance(orig_acts, int) or misc_params['mode'] == 'val':
        orig_acts = torch.zeros_like(out)
    bases = bases_module.weight
    weights = torch.sigmoid(weights_param)
    # if misc_params['mode'] == 'val':
    #     weights = (weights > misc_params['thresh']).float()
    # if misc_params['flip']:
    #     weights = 1 - weights
    m_out = (bases.T@((weights*(bases_module(orig_acts)) 
            +(1 - weights)*(bases_module(out)) ).unsqueeze(-1))).squeeze()
    return m_out

def better(metric1, metric2, metric_polarity):
    m1_better = False
    m2_better = False
    for k in metric1.keys():
        if metric_polarity[k] == 'higher':
            if metric1[k] > metric2[k]:
                m1_better = True
            elif metric1[k] <= metric2[k]:
                m2_better = True
        elif metric_polarity[k] == 'lower':
            if metric1[k] < metric2[k]:
                m1_better = True
            elif metric1[k] >= metric2[k]:
                m2_better = True
        if m1_better and m2_better:
            return None
    if m1_better:
        return True
    if m2_better:
        return False
    raise Exception("Shouldn't get here")


def add_if_better(metric, metric_list, metric_orders, weights_param, bases_modules, save_path):
    new_metric_list = []
    rounded_metric_str = dict_to_str(metric)
    rounded_metric = str_to_dict(rounded_metric_str)
    for candidate_metric in metric_list:
        is_better = better(rounded_metric, candidate_metric, metric_orders)
        if is_better is None:
            new_metric_list.append(candidate_metric)
            continue
        if is_better:
            # delete folder
            os.system(f'rm -rf {save_path+f"/{dict_to_str(candidate_metric)}"}')
        else:
            return metric_list
    os.makedirs(save_path+f'/{rounded_metric_str}', exist_ok=True)

    torch.save({'weights':weights_param, 
                'bases':dict([(k,v.state_dict()) for k,v in bases_modules.items()]),
                'metric':metric,
                }, 
               save_path+f'/{rounded_metric_str}/weights_bases_param.pt')
    # torch.save(bases_modules, save_path+f'/{dict_to_str(metric)}/bases_modules.pt')
    new_metric_list.append(rounded_metric)
    return new_metric_list

def setup_params(model, normalizer, weight_init, dataloader, store_dict, hook_names, per_token=False, use_bases=False):
    all_handles = []
    for hook_name in hook_names:
        handle = my_getattr(model, hook_name).register_forward_hook(get_hook(hook_name, read_output))
        all_handles.append(handle)
    
    weights_param = dict()
    bases_modules = dict()
    
    for imgs, masks, _ in dataloader:
        imgs = imgs.to(device)
        masks = masks.to(device)
        masks = masks[:,0:1]
        with torch.no_grad():
            _ = model(normalizer(imgs)).cpu()
        for hook_name in hook_names:
            act = store_dict[hook_name]
            _, l, b = act.shape
            weights_param[hook_name] = torch.full((1, l if per_token else 1, b), weight_init, requires_grad=True, device=device)
            bases_params = torch.nn.Linear(b, b, bias=False).to(device)
            torch.nn.init.eye_(bases_params.weight)
            if use_bases:   
                bases_modules[hook_name] = orthogonal(bases_params, orthogonal_map=None)
            else:
                bases_params.weight.requires_grad_(False)
                bases_modules[hook_name] =bases_params
        break
        
    for handle in all_handles:
        handle.remove()
    all_handles = []  
    return weights_param, bases_modules

def find_causal_neurons(model, normalizer, train_dataloader, val_dataloader, store_dict, hook_names, hyperparams,
                        weights_param=None, weight_init=0, bases_modules=None, save_path=None, save_freq=1, log_freq=2,
                        per_token=False, use_bases=False, mask_type='lm', flip=True, custom_val_metrics=None, custom_val_metrics_freq=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
        
    if weights_param is None:
        weights_param, bases_modules = setup_params(model, normalizer, weight_init, train_dataloader, store_dict, hook_names, per_token, use_bases)
    # replacement_acts = dict([(name, torch.zeros_like(wp, requires_grad=True, device=device) ) 
    #                          for name, wp in weights_param.items()])

    all_parameters = [weights_param.values()]#, replacement_acts.values()]
    if use_bases:
        all_parameters += [module.parameters() for module in bases_modules.values()]
    all_parameters += [model.head.parameters()]
    all_parameters = chain(*all_parameters)
    opt = torch.optim.AdamW(all_parameters, lr=hyperparams['lr'], betas=hyperparams['betas'])
#     opt = torch.optim.SGD(all_parameters, lr=hyperparams['lr'])
    orig_acts_dict = dict([(name, None) for name in hook_names])
    modify_fn_dict = dict([(name, None) for name in hook_names])
    misc_params = {'mode':'train', 'thresh':hyperparams['thresh']}
    for hook_name in hook_names:
        modify_fn = partial(modify_output, orig_acts_dict=orig_acts_dict, 
                                           weights_param=weights_param[hook_name], 
                                           bases_module=bases_modules[hook_name],
                                           misc_params=misc_params)
        modify_fn_dict[hook_name] = modify_fn
        my_getattr(model, hook_name).register_forward_hook(get_hook(hook_name, read_output, modify_fn=modify_fn_dict))

    my_getattr(model, "head").register_forward_hook(get_hook("features", read_input, modify_fn=None))
    # try:
    #     metric_list = [str_to_dict(x) for x in os.listdir(save_path)]
    # except FileNotFoundError:
    #     metric_list = []
    metric_orders = {'accdiff':'lower', 'lossdiff':'lower', 'l1reg':'lower'}

    if custom_val_metrics is not None:
        for hook_name in hook_names:
            orig_acts_dict[hook_name] = 0#torch.zeros(1, 1, device=device)
            #replacement_acts[hook_name]#orig_acts_dict[hook_name].mean(0, keepdims=True)
        custom_metrics = custom_val_metrics(model, normalizer)
        wandb.log(custom_metrics)

    for e in range(hyperparams['epochs']):
        misc_params['mode'] = 'train'
        dataloader = chain(train_dataloader, [(None, None, None)], val_dataloader)
        for i, (imgs, masks, labels) in enumerate(dataloader):
            if imgs is None:
                misc_params['mode'] = 'val'
                # metric_accum_dict = {'accdiff':[], 'lossdiff':[], 'l1reg':[]}
                metric_accum_dict = defaultdict(list)
                continue
            imgs = imgs.to(device)
            masks = masks.to(device)
            masks = masks[:,0:1]

            # get original logits
            for hook_name in hook_names:
                orig_acts_dict[hook_name] = None

            orig_logits = model(normalizer(imgs)).detach()
            orig_features = store_dict['features'].detach()
            orig_intermediate_acts = store_dict.copy()
            # get masked logits
            if flip:
                masks = 1-masks
            if mask_type == 'lm':
                masked_imgs = (normalizer(imgs), masks, None)
                flip_masked_imgs = (normalizer(imgs), 1-masks, None)
            elif mask_type == 'greyout':
                masked_imgs = normalizer(imgs)*(masks)
                flip_masked_imgs = normalizer(imgs)*(1-masks)
            else:
                raise Exception('Mask type not defined')
            
            flip_masked_logits = model(flip_masked_imgs)
            flip_masked_features = store_dict['features']
            
            for hook_name in hook_names:
                orig_acts_dict[hook_name] = orig_intermediate_acts[hook_name].detach()

            flip_masked_logits_intervened = model(flip_masked_imgs)
            flip_masked_features_intervened = store_dict['features']

            masked_logits_intervened = model(masked_imgs)
            masked_features_intervened = store_dict['features']

            for hook_name in hook_names:
                orig_acts_dict[hook_name] = 0#torch.zeros_like(orig_intermediate_acts[hook_name])
            
            # get loss and reg
            # loss = F.relu(F.cross_entropy(masked_logits, labels.to(device), reduction='none') 
            #               - F.cross_entropy(orig_logits, labels, reduction='none').to(device)).mean()
            loss1 = F.mse_loss(masked_features_intervened, orig_features) #info regarding masked portion is restored
            loss2 = F.mse_loss(flip_masked_features_intervened, flip_masked_features) # no info regarding masked portion is leaked 
            preds_intervened = model(normalizer(imgs))
            loss3 = F.cross_entropy(preds_intervened, labels.to(device))
            true_acc = (preds_intervened.argmax(-1).cpu() == labels).float().mean().item()
            # reg_vec = torch.cat([torch.sigmoid(w).reshape(-1) for w in weights_param.values()])
            # reg = reg_vec.abs().mean() #make the weights sparse
            total_loss = loss1 + hyperparams['gamma']*loss2 + hyperparams['beta']*loss3# + hyperparams['lambda']*reg

            clean_acc = (orig_logits.argmax(-1).cpu() == labels).float().mean().item()
            masked_acc = (masked_logits_intervened.argmax(-1).cpu() == labels).float().mean().item()

            # send to live loss plot
            if misc_params['mode'] == 'train':
                # optimize!
                opt.zero_grad()
                total_loss.backward()
                opt.step()
                if i % log_freq == 0:
                    wandb.log({'loss 1 (train)': loss1.detach().item(),
                            'loss 2 (train)': loss2.detach().item(),
                            'loss 3 (train)': loss3.detach().item(),
                            'true acc (train)': true_acc,
                            # 'acc loss (train)': acc_loss.detach().item(), 
                            # 'l1 reg (train)': reg.detach().item(),
                            # 'acc diff (train)': clean_acc - masked_acc,
                            'total loss (train)': total_loss.detach().item(),
                        })
            elif misc_params['mode'] == 'val':
                metric_accum_dict['true acc (val)'].append(true_acc)
                metric_accum_dict['acc diff (val)'].append(clean_acc - masked_acc)
                metric_accum_dict['loss 1 (val)'].append(loss1.detach().item())
                metric_accum_dict['loss 2 (val)'].append(loss2.detach().item())
                metric_accum_dict['loss 3 (val)'].append(loss3.detach().item())
                # metric_accum_dict['l1 reg (val)'].append((reg_vec).float().mean().item())

        metrics = dict([(k, np.mean(v)) for k, v in metric_accum_dict.items()])  
        
        del imgs, masks, labels, masked_imgs, flip_masked_imgs, orig_logits, masked_logits_intervened, flip_masked_logits_intervened
        del total_loss, loss1, loss2, loss3, orig_intermediate_acts#x,  reg_vec, reg
        for hook_name in hook_names:
            # orig_acts_dict[hook_name] = None
            store_dict[hook_name] = None

        if custom_val_metrics is not None and e % custom_val_metrics_freq == 0:
            for hook_name in hook_names:
                orig_acts_dict[hook_name] = 0#orig_acts_dict[hook_name].mean(0, keepdims=True)
                #replacement_acts[hook_name]#orig_acts_dict[hook_name].mean(0, keepdims=True)
            custom_metrics = custom_val_metrics(model, normalizer)
            metrics.update(custom_metrics)

        wandb.log(metrics)
        metrics = dict([(k.replace(' ','').replace('(val)', ''),v) for k,v in metrics.items()])
        metrics = dict([(k,v) for k,v in metrics.items() if k in metric_orders.keys()])
        
        if e % save_freq == 0:
            torch.save({'weights':weights_param, 
                        'bases':dict([(k,v.state_dict()) for k,v in bases_modules.items()]),
                        'metric':metrics,}, 
                        os.path.join(wandb.run.dir, f"wb_params_{e}.pt"))
        #metric_list = add_if_better(metrics, metric_list, metric_orders, weights_param, bases_modules, save_path)
    return weights_param, bases_modules

def im9_val_metrics(model, normalizer, num_workers=4, batch_size=512):
    metrics = {}
    for mode in ['original', 'mixed_rand', 'mixed_same', 'only_fg']:
        acc = get_im9_val_acc(model, normalizer, batch_size, num_workers, mode=mode)
        metrics[mode.replace('_', ' ')+' acc (val)'] = acc
    # orig_acc = get_im9_val_acc(model, normalizer, num_workers, batch_size, mode='original')
    # mr_acc = get_im9_val_acc(model, normalizer, num_workers, batch_size, mode='mixed_rand')
    # ms_acc = get_im9_val_acc(model, normalizer, num_workers, batch_size, mode='mixed_same')
    # fgonly_acc = get_im9_val_acc(model, normalizer, num_workers, batch_size, mode='fg_only')
    # metrics = {'orig_acc': orig_acc, 'mr_acc': mr_acc, 'ms_acc': ms_acc, 'fgonly_acc': fgonly_acc}
    return metrics

def get_toy_data_metrics(model, normalizer, loader):
    metrics = {}
    hits = 0
    total = 0
    for imgs, _, labels in loader:
        imgs = imgs.to(device)
        with torch.no_grad():
            logits = model(normalizer(imgs))
        hits += (logits.argmax(-1).cpu() == labels).float().sum().item()
        total += len(labels)    
    metrics['uncorr acc (val)'] = hits/total
    return metrics
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='deit_b_16')
    parser.add_argument('--dataset', type=str, default='imagenet9')
    parser.add_argument('--mask_type', type=str, default='greyout')
    parser.add_argument('--hook_names', nargs='+', default=['norm'], type=str)
    parser.add_argument('--flip', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--per_token', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--use_bases', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--use_custom_val_metrics', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--weight_init', type=float, default=-3.0)
    parser.add_argument('--lamb', type=float, default=1)
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_batches', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--custom_val_metrics_freq', type=int, default=5)
    parser.add_argument('--save_dir', type=str, default='./causal_neurons/')
    parser.add_argument('--weights_dir', type=str, default='')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Allocated {os.cpu_count()} cpus and {torch.cuda.device_count()} gpus')    
    # fg_loader = data_module.onlyfg_dataset.make_loaders(num_workers, batch_size, mode='val', shuffle_val=False)
    # bg_loader = data_module.nofg_dataset.make_loaders(num_workers, batch_size, mode='val', shuffle_val=False)
    batch_size = max(args.gpu_size*torch.cuda.device_count(), 100) if args.batch_size is None else args.batch_size

    if args.dataset == 'imagenet9':
        data_module = ImageNet9DataModule(get_masks=True)
        data_module.setup()
        orig_train_loader, orig_val_loader = data_module.orig_dataset.make_loaders(args.num_workers,batch_size, mode='val', shuffle_val=True, subset_ratio=0.875, rand_split=True)
        print(f"Lengths of dataloaders with batch size {batch_size}: train-{len(orig_train_loader)}, val-{len(orig_val_loader)}")
        lightning_model = ImageNet9Predictor.load_from_checkpoint('./models/imagenet9/mtype-deit_b_16_num_classes-9_lr-1e-05_weight_decay-1e-06/epoch=01-step=1200-orig_val_acc=0.99.ckpt')
        custom_val_metrics = partial(im9_val_metrics, num_workers=args.num_workers, batch_size=batch_size) if args.use_custom_val_metrics else None
        train_loader = orig_train_loader
        val_loader = orig_val_loader
    elif args.dataset == 'toy_datasets_2' or args.dataset == 'toy_datasets':
        data_module = ToyDataModule(root_dir=f'/cmlscratch/sriramb/{args.dataset}/', batch_size=batch_size, get_masks=True)
        data_module.setup()
        corr_train_loader = data_module.train_dataloader()
        corr_val_loader, uncorr_loader = data_module.val_dataloader()
        lightning_model = ToyDatasetPredictor.load_from_checkpoint(f'./models/{args.dataset}/mtype=deit_b_16_num_classes=2._lr=0.000005_weight_decay=0.000003/last.ckpt')
        train_loader = corr_train_loader
        val_loader = corr_val_loader
        custom_val_metrics = partial(get_toy_data_metrics, loader=uncorr_loader) if args.use_custom_val_metrics else None
    else:
        raise NotImplementedError

    model = lightning_model.model.to(device)
    normalizer = lightning_model.normalizer

    hyperparams = {
                    'lr':args.lr,
                    'betas':(0.9, 0.999),
                    'lambda': args.lamb,
                    'gamma': args.gamma,
                    'beta': args.beta,
                    'epochs':args.epochs+1,
                    'thresh':0.05,
                    'batch_size':batch_size,
                }

    hook_name_str = "_".join(args.hook_names)
    save_path = args.save_dir + f'/{args.model}_{args.dataset}_{args.mask_type}/flip-{args.flip}/{hook_name_str}/'
    # param_path = save_path + f'/{dict_to_str(hyperparams)}'

    try:
        weights_bases_param = torch.load(save_path+f'{args.weights_dir}/weights_bases_param.pt')
        weights_param, bases_modules_sd = weights_bases_param['weights_param'], weights_bases_param['bases_modules']
        _, bases_modules = setup_params(model, normalizer, args.weight_init, orig_train_loader, activation, args.hook_names, args.per_token, args.use_bases)
        for k, v in bases_modules.items():
            v.load_state_dict(bases_modules_sd[k])
    except Exception as e:
        print(e)
        print('No weights found, initializing from scratch')
        weights_param = None
        bases_modules = None

    run = wandb.init(
            project=f"find-causal-neurons-{args.model}_{args.dataset}_{args.mask_type}_flip-{args.flip}_{hook_name_str}",
            config=hyperparams
            )
    
    #_{args.flip}_{args.per_token}_{args.use_bases}_{args.weight_init}_{args.lambda}_{args.lr}_{args.epochs}_{args.batch_size}_{args.num_batches}_{args.num_workers}_{args.gpu_size}_{args.seed}
    weights_param, bases_modules = find_causal_neurons(model, normalizer, train_loader, val_loader, activation, 
                                                        args.hook_names, hyperparams,
                                                        weights_param=weights_param, 
                                                        weight_init=args.weight_init, 
                                                        bases_modules=bases_modules,
                                                        save_path=save_path, save_freq=args.save_freq,
                                                        per_token=args.per_token, use_bases=args.use_bases,
                                                        mask_type=args.mask_type, flip=args.flip, 
                                                        custom_val_metrics=custom_val_metrics, custom_val_metrics_freq=args.custom_val_metrics_freq)