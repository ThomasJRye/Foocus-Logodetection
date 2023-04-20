import config
import torch
from utils import myOwnDataset
from utils import get_transform, collate_fn
import torchvision.transforms as transforms
from torch.utils.data import WeightedRandomSampler
from torchvision.models.detection import fasterrcnn_resnet50_fpn

def train_model(model, device, transforms = None):

    if transforms is None:
        # Create own Dataset
        training_dataset = myOwnDataset(
            root=config.data_dir, annotation=config.train_coco, transforms=get_transform()
        )
    else:
        # Create own Dataset
        training_dataset = myOwnDataset(
            root=config.data_dir, annotation=config.train_coco, transforms=transforms
        )
        

    # Create own Dataset
    training_dataset = myOwnDataset(
        root=config.data_dir, annotation=config.train_coco, transforms=get_transform()
    )

    # Training DataLoader
    training_loader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=config.train_batch_size,
        shuffle=config.train_shuffle_dl,
        num_workers=config.num_workers_dl,
        collate_fn=collate_fn,
    )

    # Parameters
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay
    )

    len_dataloader = len(training_loader)
    loss_per_epoch = []

    # Training
    for epoch in range(config.num_epochs):
        print(f"Epoch: {epoch}/{config.num_epochs}")
        model.train()
        i = 0

        sum_loss = 0
        for imgs, annotations in training_loader:
            i += 1
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]

            # Check if there are any boxes in the annotations
            has_boxes = any(anno['boxes'].size(0) > 0 for anno in annotations)

            # If there are no boxes in any of the images in the batch, skip the iteration
            if not has_boxes:
                continue

            loss_dict = model(imgs, annotations)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            print(f"Iteration: {i}/{len_dataloader}, Loss: {losses}")
            sum_loss += losses

        # Print average loss for epoch
        print(f"Average loss for epoch: {sum_loss/len_dataloader}")

        loss_per_epoch.append(sum_loss/len_dataloader)