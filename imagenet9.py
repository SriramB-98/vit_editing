
import warnings
import pickle
from tools.datasets import ImageNet, ImageNet9
import json
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

original = ImageNet9('/cmlscratch/sriramb/imagenet9/original')
mixed_rand = ImageNet9('/cmlscratch/sriramb/imagenet9/mixed_rand')
mixed_same = ImageNet9('/cmlscratch/sriramb/imagenet9/mixed_same')
only_fg = ImageNet9('/cmlscratch/sriramb/imagenet9/only_fg')
map_to_in9 = {}
with open('/nfshomes/sriramb/projects/sal_imagenet_ood/in_to_in9.json', 'r') as f:
    map_to_in9.update(json.load(f))
with open('/cmlscratch/mmoayeri/analysis_causal_imagenet/meta/ftr_types/by_class_dict.pkl', 'rb') as f:
    by_class_dict = pickle.load(f)

# core = []
# spur = []
# for i, j in map_to_in9.items():
#     if j != -1:
#         core.extend(by_class_dict[int(i)]['core'])
#         spur.extend(by_class_dict[int(i)]['spurious'])
# core = set(core)
# spur = set(spur)

# only_core = list(core - spur)
# no_only_spur = list(core)

def map_preds_to_in9(preds):
    in9_preds = torch.zeros(*preds.shape)
    for i, l in enumerate(preds):
        in9_preds[i] = map_to_in9[str(l.item())]
    return in9_preds



def get_im9_val_acc(model, normalizer, batch_size=512, num_workers=8, mode='original', use_im9_mapper=False):
    if use_im9_mapper:
        im9_mapper = torch.zeros(1000, 9)
        for ic, i9c in map_to_in9.items():
            if i9c == -1:
                continue
            im9_mapper[int(ic)][i9c] = 1
        im9_mapper = im9_mapper.to(device)

    if mode == 'original':
        loader = original.make_loaders(workers=num_workers, batch_size=batch_size, shuffle_val=False)
    elif mode == 'mixed_same':
        loader = mixed_same.make_loaders(workers=num_workers, batch_size=batch_size, shuffle_val=False)
    elif mode == 'mixed_rand':
        loader = mixed_rand.make_loaders(workers=num_workers, batch_size=batch_size, shuffle_val=False)
    elif mode == 'only_fg':
        loader = only_fg.make_loaders(workers=num_workers, batch_size=batch_size, shuffle_val=False)
        
    samples = 0
    acc = 0
    model.eval()
    for i, (imgs, labels) in enumerate(loader):
        labels = labels.to(device)
        imgs = imgs.to(device)
        with torch.no_grad():
            preds = model(normalizer(imgs))
            if use_im9_mapper:
                preds = torch.softmax(preds, dim=-1)@im9_mapper
        acc += (preds.argmax(-1)==labels).float().sum()
#         acc += (map_preds_to_in9(preds.argmax(-1))==labels.cpu()).float().sum()
        samples += len(labels)
#     print(f'val acc:{acc/samples}')
    model.train()
    return (acc/samples).item()


# def finetune_model_im9(model, dataset, val_loader, frozen_params, its_step=10_000, num_steps=10, subset_ratio=1, hyperparams={'lr':1e-3, 'bs':128, }):
#     num_its = its_step*num_steps
#     loader = dataset.make_loaders(batch_size=hyperparams['bs'], workers=4*torch.cuda.device_count(), mode='train', subset_ratio=subset_ratio, shuffle_val=True)
#     print(f"Total number of samples: {len(loader)*hyperparams['bs']}")
#     model.train()
#     model.set_req_grads(*frozen_params)
#     opt = torch.optim.Adam(model.parameters(), lr=hyperparams['lr'], eps=hyperparams['eps'])
#     loss_fn = torch.nn.NLLLoss()
#     it = 0
#     val_accs = []
#     train_losses = []
#     max_val_acc = 0
#     strikes = 0
#     while True:
#         for _, (imgs, labels) in enumerate(loader):
#             if it%its_step == 0:
#                 val_accs.append(get_val_acc(model, val_loader))
#             imgs, labels = imgs.to(device), labels.to(device)
#             preds = torch.softmax(model(imgs), dim=1)@im9_mapper
#             logprobs = torch.log(preds/torch.sum(preds, dim=1, keepdims=True))
#             loss = loss_fn(logprobs, labels)
#             opt.zero_grad()
#             loss.backward()
#             try:
#                 opt.step()
#             except Exception as e:
#                 print(e)
#                 return train_losses, val_accs, model
#             train_losses.append(loss.detach().item())
#             if it%its_step == 0:
#                 print(f'training loss at {it}: {loss}')
#             it += 1
#             if it > num_its or strikes > 3:
#                 model.eval()
#                 return train_losses, val_accs, model