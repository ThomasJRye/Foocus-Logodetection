from multiprocessing import freeze_support
import torch
import config
from utils import (
    collate_fn,
    get_transform,
    myOwnDataset,
)
import model
import json
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from training import train_model
from eval import evaluate_model
import datetime, os
# Load json file
with open(config.test_coco) as f:
    json_data = json.load(f)

if __name__ == '__main__':
    freeze_support()

    # Crate test Dataset
    testing_dataset = myOwnDataset(
        root=config.data_dir, annotation=config.test_coco, transforms=get_transform()
    )

    # Testing DataLoader
    testing_loader = torch.utils.data.DataLoader(
        testing_dataset,
        batch_size=config.test_batch_size,
        shuffle=config.train_shuffle_dl,
        num_workers=config.num_workers_dl,
        collate_fn=collate_fn,
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # Instantiate and train the model
    notransform_model = model.get_model(device=device, model_name='v2')

    writer = SummaryWriter()

    # Train the model
    train_model(notransform_model, device, None, writer)

    # Close the SummaryWriter
    writer.close()

    # Evaluate the model
    evaluate_model(notransform_model, device, testing_loader, "notrans")

    # # Load the TensorBoard notebook extension
    # %load_ext tensorboard

    # # Start TensorBoard
    # %tensorboard --logdir logs
