# path to your own data and coco file
data_dir = "Logodetection.v3i.coco/"
roboflow = True

# Batch size
train_batch_size = 7
test_batch_size = 7




# Params for dataloader
train_shuffle_dl = True
num_workers_dl = 4

# Params for training
lr_step_size = 2
lr_gamma = 0.6

# Two classes; Only target class or background
num_classes = 32
num_epochs = 12

lr = 0.01
momentum = 0.9
weight_decay = 0.005