# utility packages
import sys, os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import json

# pytorch
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

# plotting
import matplotlib.pyplot as plt

# attack modules
import OP_attack_fast 
import OnePixelAttack as OP_attack_original

# custom dataset for loading images
class ImagenetDataset(Dataset):
    def __init__(self, image_dir, image_names, transform=None):
        self.image_dir = image_dir
        self.image_names = image_names
        self.transform = transform
        self.len = len(self.image_names)
        
    def get_class_label(self, image_name):
        # your method here
        y = label2idx[image_name[:9]]
        return torch.tensor(y).long()
        
    def __getitem__(self, index):
        image_path = os.path.join(image_dir, self.image_names[index])
        x = T.ToTensor()(Image.open(image_path).convert('RGB'))
        y = self.get_class_label(image_names[index])
        if self.transform is not None:
            x = self.transform(x)
        return x, y
    
    def __len__(self):
        return self.len

def perturb_image(xs, img):
    xs = xs.reshape(-1, 5)
    xs = xs.astype(int)
    adv_example = img.clone()
    for pixel in xs:
        adv_example[:, 0, pixel[0], pixel[1]] = pixel[2]/255.0
        adv_example[:, 1, pixel[0], pixel[1]] = pixel[3]/255.0
        adv_example[:, 2, pixel[0], pixel[1]] = pixel[4]/255.0
    return adv_example


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--pixels", default=20, help="number of pixels for attack, type:int", type=int)
    args = ap.parse_args()

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load pretrained model
    resnet18 = models.resnet18(pretrained=True)
    resnet18.to(device).eval()

    # load Imagenet labels encodings
    idx2label = []
    cls2label = {}
    label2idx = {}
    with open("imagenet/imagenet_class_index.json", "r") as read_file:
        class_idx = json.load(read_file)
        idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
        cls2label = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}
        label2idx = {class_idx[str(k)][0]: k for k in range(len(class_idx))}

    # create dataset
    image_dir = "imagenet/test-images"
    image_names = sorted(os.listdir("imagenet/test-images"))
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), normalize])
    inverse_transform = T.Compose([T.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                   T.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ])])

    dataset = ImagenetDataset(image_dir, image_names, transform=None)

    # Run ResNet18 without attacks
    print("Testing network without attacks...")
    correct = 0
    for x, y in tqdm(dataset, total=len(dataset)):
        x = x.unsqueeze(0).to(device)
        x = transform(x)
        y_pred = torch.argmax(resnet18(x))
        if y_pred == y:
            correct += 1
    print(f"Accuracy of network without attack: {correct/len(dataset)*100}")

    # Run OP attack fast on whole dataset
    print("Testing network with fast OP attack...")
    op_attack_fast = OP_attack_fast.OnePixelAttack(resnet18, transform, device)
    correct = 0
    total = 0
    nfevs = 0
    bar = tqdm(dataset, total=len(dataset))
    for x, y in bar:
        success, perturbation, nfev, attack_image = op_attack_fast.attack(x.clone().unsqueeze(0), y.view(1), pixels=args.pixels)#, verbose=True)
        correct += (1-success)
        total += 1
        nfevs += nfev
        with torch.no_grad():
            y_pred = torch.argmax(resnet18(attack_image.to(device)))
        bar.set_description(f"Accuracy: {correct/total*100}, mean no. calls to network: {nfevs/total}, True class: {y}, Pred class: {y_pred}")
    print(f"Accuracy: {correct/total*100}, mean no. calls to network: {nfevs/total}")

    # Run OP attack original on whole dataset
    print("Testing network with original OP attack...")
    op_attack_original = OP_attack_original.OnePixelAttack(resnet18, pixels=args.pixels, steps=75, popsize=400, inf_batch=128)
    correct = 0
    total = 0
    nfevs = 0
    bar = tqdm(dataset, total=len(dataset))
    for x, y in bar:
        attack_image, nfev = op_attack_original.forward(x.clone().unsqueeze(0), y.view(1))
        total += 1
        nfevs += nfev
        with torch.no_grad():
            y_pred = torch.argmax(resnet18(attack_image.to(device)))
        if y_pred == y:
            correct += 1
        bar.set_description(f"Accuracy: {correct/total*100}, mean no. calls to network: {nfevs/total}, True class: {y}, Pred class: {y_pred}")
    print(f"Accuracy: {correct/total*100}, mean no. calls to network: {nfevs/total}")


