import torch
from collections import defaultdict
import detect_utils
from coco_names import COCO_INSTANCE_CATEGORY_NAMES

def evaluate_model(model, device, testing_loader):
    model.eval()

    # Evaluate the model
    with torch.no_grad():
        category_correct_boxes = defaultdict(int)
        category_correct_labels = defaultdict(int)
        category_total_boxes = defaultdict(int)

        for imgs, annotations in testing_loader:
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]

            for idx, image in enumerate(imgs):
                image_np = image.cpu().numpy().transpose(1, 2, 0)

                boxes, pred_classes, labels = detect_utils.predict(image_np, model, device, 0.4)
                print("labels: ")
                print(pred_classes)

                gt_boxes = annotations[idx]['boxes'].cpu().numpy()
                gt_labels = annotations[idx]['labels'].cpu().numpy()

                correct_labels = []

                for label_index in gt_labels:
                    if (label_index == 32):
                        label_index = 0
                    correct_labels.append(COCO_INSTANCE_CATEGORY_NAMES[label_index])

                print(correct_labels)
                for gt_box, gt_label in zip(gt_boxes, correct_labels):
                    category_total_boxes[gt_label] += 1
                    iou_threshold = 0.5

                    for pred_box, pred_label in zip(boxes, pred_classes):
                        iou = detect_utils.calculate_iou(gt_box, pred_box)

                        if iou > iou_threshold:
                            category_correct_boxes[gt_label] += 1
                            if gt_label == pred_label:
                                category_correct_labels[gt_label] += 1
                            break

        # Calculate and print accuracy for each category
        for category in category_total_boxes.keys():
            box_accuracy = category_correct_boxes[category] / category_total_boxes[category]
            label_accuracy = category_correct_labels[category] / category_total_boxes[category]

            print(f"Category: {category}")
            print(f"Bounding Box Accuracy: {box_accuracy * 100:.2f}%")
            print(f"Label Accuracy: {label_accuracy * 100:.2f}%")

