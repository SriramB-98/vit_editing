from functools import partial
import torch
from collections import defaultdict

def my_getattr(m, x, y=None):
    xis = x.split('.')
    t = m
    for xi in xis:
        if xi.isnumeric():
            t = t[int(xi)]
        else:
            t = getattr(t, xi, y)
    return t

def get_hook(name, read_fn=None, modify_fn=None):
    # the hook signature
    def hook(model, input, output):
        if read_fn is not None:
            read_fn(input, output, name)
        if modify_fn is not None:
            mf = modify_fn[name] if isinstance(modify_fn, dict) else modify_fn
            output = mf(input, output, name)
#             read_fn(input, output, name+"_after")
            return output
    return hook

activation = {}
def read_output(inp, out, name):
    if isinstance(out, tuple):
        activation[name] = out[0].detach()#.cpu()
    else:
        activation[name] = out.detach()#.cpu()
    if name == 'encoder' or name == 'norm':
        activation[name] = activation[name][:,0:1]
    return

def read_input(inp, out, name):
    if isinstance(inp, tuple):
        activation[name] = inp[0].detach()#.cpu()
    else:
        activation[name] = inp.detach()#.cpu()
    if name == 'encoder' or name == 'norm':
        activation[name] = activation[name][:,0:1]
    return

def get_intermediate_acts(model, normalizer, dataloader, hook_names, reduction='mean'):
    if reduction == 'mean':
        reduction = partial(torch.mean, dim=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for hook_name in hook_names:
        my_getattr(model, hook_name).register_forward_hook(get_hook(hook_name, read_fn=read_output))
    all_acts = dict([(hook_name, list()) for hook_name in hook_names])
    for batch in dataloader:
        imgs = batch[0].to(device)
        with torch.no_grad():
            _ = model(normalizer(imgs)).cpu()
            for hook_name in hook_names:
                all_acts[hook_name].append(reduction(activation[hook_name].cpu()))
    for hook_name in hook_names:
        all_acts[hook_name] = reduction(torch.stack(all_acts[hook_name], dim=0)).unsqueeze(0)
    return all_acts

# def get_intermediate_acts(model, normalizer, dataloader, store_dict, label_names, mask_type='lm'):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # all_acts = defaultdict(lambda: [list() for _ in label_names])
    # all_orig_hits = [0 for _ in label_names]
    # all_fg_hits = [0 for _ in label_names]
    # all_bg_hits = [0 for _ in label_names]
    # totals = [0 for _ in label_names]
    # for imgs, masks, labels in dataloader:
    #     imgs = imgs.to(device)
    #     masks = masks.to(device)
    #     masks = masks[:,0:1]
    #     with torch.no_grad():
    #         orig_logits = model(normalizer(imgs)).cpu()
    #         orig_acts = store_dict.copy()
            
    #         if mask_type == 'lm':
    #             bg_imgs = (normalizer(imgs), 1-masks, None)
    #             fg_imgs = (normalizer(imgs), masks, None)
    #         elif mask_type == 'greyout':
    #             bg_imgs = normalizer(imgs)*(1-masks)
    #             fg_imgs = normalizer(imgs)*(masks)
    #         else:
    #             raise Exception('Mask type not defined')
    #         bg_logits = model(bg_imgs).cpu()
    #         bg_acts = store_dict.copy()
    #         fg_logits = model(fg_imgs).cpu()
    #         fg_acts = store_dict.copy()
        
    #     orig_hits = orig_logits.argmax(-1) == labels
    #     fg_hits = fg_logits.argmax(-1) == labels
    #     bg_hits = bg_logits.argmax(-1)  == labels
    #     for i in range(len(label_names)):
    #         for hook_name in orig_acts.keys():
    #             oa = orig_acts[hook_name][labels==i]
    #             ba = bg_acts[hook_name][labels==i]
    #             fa = fg_acts[hook_name][labels==i]
    #             all_acts[hook_name][i].append(torch.stack([oa, ba, fa], dim=1))
    #         all_orig_hits[i] += torch.sum(orig_hits[labels==i]).item()
    #         all_fg_hits[i] += torch.sum(fg_hits[labels==i]).item()
    #         all_bg_hits[i] += torch.sum(bg_hits[labels==i]).item()
    #         totals[i] += (labels==i).sum().item()
    # for hook_name in all_acts.keys():
    #     all_acts[hook_name] = [torch.cat(acts, dim=0) for acts in all_acts[hook_name]]
    # return all_acts, all_orig_hits, all_fg_hits, all_bg_hits, totals