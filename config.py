import os

class Config:
    epochs  = 1
    n_class = 100
    base_lr = 1e-4
    batch_size = 256
    num_epochs = 15
    img_size   = 224
    workers    = 1

    # DataSet
    img_dir  = './CIFAR100'  # set with yours
    download = not os.path.exists(img_dir)

    # Valid
    logsteps = 100