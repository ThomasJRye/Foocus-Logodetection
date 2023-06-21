# path to your own data and coco file
data_dir = "bigData1/"
train_coco = "bigData1/train.json"
test_coco = "bigData1/test.json"


# Batch size
train_batch_size = 7
test_batch_size = 7




# Params for dataloader
train_shuffle_dl = True
num_workers_dl = 4

# Params for training
lr_step_size = 3
lr_gamma = 0.1

# Two classes; Only target class or background
num_classes = 157
num_epochs = 12

lr = 0.005
momentum = 0.9
weight_decay = 0.005