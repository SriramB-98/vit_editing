from sklearn import svm
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
import torchvision.transforms as transforms
import torch 
import glob
from PIL import Image


local_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_waterbirds_loader(bs=128, num_workers=4):
    waterbirds_dataset = get_dataset("waterbirds", root_dir='/cmlscratch/mmoayeri/data')
    # an alternate transformation can also be used -- couldnt find the exact original size
    transform = transforms.Compose([transforms.Resize(224), 
                                    transforms.CenterCrop(224), 
                                    transforms.ToTensor()
                                   ])

    waterbirds_train_dset = waterbirds_dataset.get_subset("train", transform=transform)
    waterbirds_test_dset = waterbirds_dataset.get_subset("test", transform=transform)

    img_list = sorted(glob.glob('/cmlscratch/sriramb/places365_val/*.jpg'))
    with open('/cmlscratch/sriramb/places365_val/places365_val.txt') as fp:
        label_list = [int(x.split()[1]) for x in fp.readlines()]

    target_places = [
        ['bamboo_forest', 'forest/broadleaf'],  # Land backgrounds
        ['ocean', 'lake/natural']]              # Water backgrounds

    target_ids = [
        [36, 150],  # Land backgrounds
        [243, 205]] # Water backgrounds 

    land_imgs, water_imgs = [], []
    for img_path, label in zip(img_list, label_list):
        if label in target_ids[0]:
            land_imgs.append(Image.open(img_path))
        elif label in target_ids[1]:
            water_imgs.append(Image.open(img_path))

    # land_imgs[0].save("land_imgs.gif", format="GIF", append_images=land_imgs, save_all=True, duration=500, loop=0)        
    # water_imgs[0].save("water_imgs.gif", format="GIF", append_images=water_imgs, save_all=True, duration=500, loop=0)  
    waterbirds_train_loader = get_train_loader("standard", waterbirds_train_dset, batch_size=bs, num_workers=num_workers)
    waterbirds_test_loader = get_eval_loader("standard", waterbirds_test_dset, batch_size=bs, num_workers=num_workers)
    
    return waterbirds_train_loader, waterbirds_test_loader, water_imgs, land_imgs 
    



def getActivation(activation, name, out=True):
    # the hook signature
    def hook(model, inp, output):
        if out:
            activation[str(inp[0].device)] = output
        else:
            activation[str(inp[0].device)] = inp
    return hook

def get_model_features(model, x):
    try:
        return model.get_features(x).cpu()
    except Exception as e:
        activation = {}
        try:
            hook = model.fc.register_forward_hook(getActivation(activation, 'target', out=False))
        except:
            hook = model.heads.register_forward_hook(getActivation(activation, 'target', out=False))
        _ = model.forward(x)
        all_acts = torch.cat([activation[f'cuda:{i}'][0].cpu() for i in range(len(activation))], dim=0)
        return all_acts

def get_all_features(model, loader):
    img_features = []
    img_labels = []
    bg_labels = []
    for i, (imgs, labels, meta) in enumerate(loader):
        imgs = imgs.to(local_device)
        with torch.no_grad():
            features = get_model_features(model, imgs).detach().cpu()
        img_features.append(features)
        img_labels.append(labels)
        bg_labels.append(meta[:,0])

    bg_labels = torch.cat(bg_labels, 0).cpu().numpy()
    img_labels = torch.cat(img_labels, 0).cpu().numpy()
    img_features = torch.cat(img_features, 0).cpu().numpy()
    return bg_labels, img_labels, img_features
    
def get_waterbirds_metrics(model, device, batch_size=512, num_workers=8):
    waterbirds_train_loader = get_train_loader("standard", waterbirds_train_dset, batch_size=batch_size, num_workers=num_workers)

    waterbirds_test_loader = get_train_loader("standard", waterbirds_test_dset, batch_size=batch_size, num_workers=num_workers)

    model.eval()
    model.to(local_device)
    metrics = dict([])
    train_bg_labels, train_img_labels, train_img_features = get_all_features(model, waterbirds_train_loader)
    test_bg_labels, test_img_labels, test_img_features = get_all_features(model, waterbirds_test_loader)
    clf = svm.SVC()
    clf.fit(train_img_features, train_img_labels)
    metrics["Train acc (img)"] = clf.score((train_img_features), (train_img_labels))
    metrics["Test acc (img)"] = clf.score((test_img_features), (test_img_labels))

    clf = svm.SVC()
    clf.fit(train_img_features, train_bg_labels)
    metrics["Train acc (bg)"] = clf.score(train_img_features, train_bg_labels)
    metrics["Test acc (bg)"] = clf.score(test_img_features, test_bg_labels)
    model.train()
    model.to(device)
    return metrics
     