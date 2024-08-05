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
Pytorch Optimization Step By Step
First, We considered increasing the number of workers, but this is not good choice because our system not good
Second, We thought about increasing the batch size, but we're already using full memory capacity.
Finally, We are able to change pytorch memory format. because cudnn only supports channel last format 
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
dataloader  = DataLoader(dataset, batch_size=configs.batch_size, shuffle=True, num_workers=configs.workers)


# Model
model = make_model(configs).cuda().to(memory_format=torch.channels_last)
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
        with torch.autograd.profiler.emit_nvtx():
            nvtx.range_start(f"{epoch} epochs")

            nvtx.range_push("Data Load")
            for X,y in loader:

                X = X.cuda().to(memory_format=torch.channels_last)
                y = y.type(torch.long).cuda()
                nvtx.range_pop()

                nvtx.range_push("Batch")
                with torch.autocast(device_type="cuda"):
                    pred_y = model(X)
                    acc_lst.append(Accuracy(pred_y, y).item())      
                    loss   = loss_fn(pred_y, y)
                    loss_lst.append(loss.item())

                    loss.backward()
                    optimizer.step()
                    pred_y = model(X)
                nvtx.range_pop() 

                # if global_step % config.logsteps == 0:
                #     print(f"{global_step} Loss :{np.mean(loss_lst):.4f} | Accuracy : {np.mean(acc_lst):.4f}")

                if global_step == 99:
                    sys.exit(1)


                global_step += 1
            nvtx.range_end()


if __name__ == "__main__":

    train(model, dataloader, configs)
    pass