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

def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    ### TRAINING

    # Instantiate and train the model
    notransform_model = model.get_model(device=device)

    # Initialize tensorboard log, default location is runs/{date}
    writer = SummaryWriter()

    # Train the model
    train_model(notransform_model,"bigData2", device, transforms=None, writer=writer)

    # save trained model
    model_path = os.path.join(writer.log_dir, "last.pth")

    # Save the model
    print(f'Saving model to {model_path}')
    torch.save(notransform_model.state_dict(), model_path)

    # Close the SummaryWriter
    writer.close()

    ### EVALUATION

    # Load json file
    with open(config.test_coco) as f:
        json_data = json.load(f)

    # Create test Dataset
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

    # Evaluate the model
    evaluate_model(notransform_model, device, testing_loader, "notrans", print_results=True)

if __name__ == '__main__':
    main()
