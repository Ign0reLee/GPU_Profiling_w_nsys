import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# load resnet18 with the pre-trained weights
from torchvision import models
from torchvision import datasets, transforms as T
from torch.utils.data import DataLoader
from utils import *
from config import *


from torch.cuda import nvtx

r"""
Using NVTX Default
Just using range_push and range_pop
Even every event can available
but it is too fool
"""

configs   = Config()
transform = T.Compose([
                        T.Resize(256),
                        T.RandomResizedCrop(configs.img_size),
                        T.RandomHorizontalFlip(),
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),      
                    ])

dataset     = datasets.CIFAR100(configs.img_dir, download=configs.download, train=True, transform=transform)
dataloader  = DataLoader(dataset, batch_size=configs.batch_size, shuffle=True, num_workers=4)


# Model
model = make_model(configs).cuda()

# set optimizer
params     = [p for p in model.parameters() if p.requires_grad == True]
optimizer  = optim.AdamW(params, lr= configs.base_lr)
loss_fn    = nn.CrossEntropyLoss()

count_parameters(model)

def train(model, loader, config):
    global_step  = 0
    loss_lst     = []
    acc_lst      = []
    for epoch in range(config.epochs):
        with torch.autograd.set_detect_anomaly(True):

            with torch.autograd.profiler.emit_nvtx():
                with torch.autocast(device_type="cuda"):
                    nvtx.range_start(f"{epoch} epochs")
                    nvtx.range_push(f"Data Loading")
                    for X,y in loader:
                        nvtx.range_pop()
                        nvtx.range_push("Copy to device")
                        X = X.cuda()
                        y = y.type(torch.long).cuda()

                        nvtx.range_pop()
                        nvtx.range_push("Run Main Model")
                        pred_y = model(X)
                        nvtx.range_pop()

                        nvtx.range_push("Calculate ACC")
                        acc_lst.append(Accuracy(pred_y, y).item())
                        nvtx.range_pop()

                        nvtx.range_push("Calculate Loss")
                        loss   = loss_fn(pred_y, y)
                        loss_lst.append(loss.item())
                        nvtx.range_pop()

                        nvtx.range_push("Backwar pass and Optimizer step")
                        loss.backward()
                        optimizer.step()
                        nvtx.range_pop()
                        
                        nvtx.range_push("Logging")
                        if global_step % config.logsteps == 0:
                            print(f"{global_step} Loss :{np.mean(loss_lst):.4f} | Accuracy : {np.mean(acc_lst):.4f}")
                        nvtx.range_pop()

                        global_step += 1

                        sys.exit(1)
                    nvtx.end_range()


if __name__ == "__main__":

    train(model, dataloader, configs)
    pass