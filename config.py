# path to your own data and coco file
data_dir = "dataset/data"
train_coco = "dataset/train.json"
test_coco = "dataset/train.json"


# Batch size
train_batch_size = 1
test_batch_size = 1
# Params for dataloader
train_shuffle_dl = True
num_workers_dl = 4

# Params for training

# Two classes; Only target class or background
num_classes = 31
num_epochs = 10

lr = 0.005
momentum = 0.9
weight_decay = 0.005