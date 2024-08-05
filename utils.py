from prettytable import PrettyTable
import os
import torch
import torch.nn as nn
import torch.optim as optim

import torch.functional as F

# from torch.cuda import nvtx
import nvtx

# load resnet18 with the pre-trained weights
from torchvision import models
from torchvision import datasets, transforms as T

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

@nvtx.annotate("Make Model")
def make_model(config):
    model = models.vgg16(pretrained=True)
    freeze_model(model)
    num_ftrs = model.classifier[0].in_features
    model.classifier = nn.Sequential(
                                        nn.Linear(num_ftrs, 2048),
                                        nn.ReLU(inplace=False),
                                        nn.Dropout(inplace=False),
                                        nn.Linear(2048, 2048),
                                        nn.ReLU(inplace=False),
                                        nn.Dropout(inplace=False),
                                        nn.Linear(2048, config.n_class)                                 
                                    )
    return model

# @nvtx.annotate("Calculate ACC")
def Accuracy(pred, gt):
    pred = pred.softmax(dim=-1).argmax(dim=-1)
    acc  = torch.sum(pred == gt) / len(gt)
    return acc
    

class nvWrite:
    def __init__(self, annot="GPU"):
        self.annot = annot
    
    def __enter__(self):
        nvtx.push_range(self.annot)
    
    def __exit__(self, type, value, trackback):
        nvtx.pop_range()