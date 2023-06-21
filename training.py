import config
from detect_utils import predict
import torch
from utils import myOwnDataset
from utils import get_transform, collate_fn
import torchvision.transforms as transforms
from torch.utils.data import WeightedRandomSampler
from torchvision.models.detection import fasterrcnn_resnet50_fpn


def train_model(model, device, transforms=None, writer=None):
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

    print("training from: " + config.train_coco)

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

    # Add learning rate scheduler
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma)

    len_dataloader = len(training_loader)
    loss_per_epoch = []

    model.train()

    # Training
    for epoch in range(config.num_epochs):
        ### epoch start ###
        i = 0
        sum_loss = 0
        print(f"Epoch {epoch}, learning_rate = {optimizer.param_groups[0]['lr']}")
        correct_predictions = 0
        writer.add_scalar(tag='train/learning_rate', scalar_value=optimizer.param_groups[0]['lr'],
                          global_step=epoch * len(training_loader) + i)
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

            sum_loss += losses

            #print(f"Epoch: {epoch}/{config.num_epochs}, batch {i}/{len_dataloader}, sum_loss = {losses.item()}")

            # Write loss to tensorboard
            if writer is not None:
                for k, v in loss_dict.items():
                    if isinstance(v, torch.Tensor):
                        v = v.item()
                    writer.add_scalar(tag=f'train/{k}', scalar_value=v, global_step=epoch * len(training_loader) + i)


        # Calculate class accuracy
        model.eval()
        with torch.no_grad():
            for imgs, annotations in training_loader:
                imgs = list(img.to(device) for img in imgs)
                annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
                for idx, image in enumerate(imgs):
                    image_np = image.cpu().numpy().transpose(1, 2, 0)
                    
                    # Check if there are any boxes in the annotations
                    has_boxes = any(anno['boxes'].size(0) > 0 for anno in annotations[idx:idx+1])

                    # If there are no boxes in the image, skip the iteration
                    if not has_boxes:
                        continue

                    # Call the predict function to get the bounding boxes, class names, and labels.
                    boxes, classes, predicted_labels = predict(image_np, model, device, 0.5)

                    gt_labels = annotations[idx]["labels"]
                    unique_predicted_labels = list(set(predicted_labels))
                    found = []
                    
                    for pred_label in unique_predicted_labels:
                        if pred_label in unique_predicted_labels and pred_label not in found:
                            correct_predictions+=1
                            found.append(pred_label)
                                
                    total_predictions = len(unique_predicted_labels)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        print(f"Class accuracy for epoch {epoch}: {accuracy * 100:.2f}%")
        model.train()
        ### epoch end ###

        # Update learning rate scheduler after epoch
        lr_scheduler.step()

        # Print average loss for epoch
        print(f"Average loss for epoch: {sum_loss / len_dataloader}")
        loss_per_epoch.append(sum_loss / len_dataloader)
    # Save the trained weights.
    model_weights_path = './models/model_weights.pth'
    torch.save(model.state_dict(), model_weights_path)
