import numpy as np
import torch
from collections import defaultdict
import detect_utils
from coco_names import COCO_INSTANCE_CATEGORY_NAMES
import csv
import cv2
import os
import torchvision.transforms as transforms
import pandas as pd

def evaluate_model(model, device, testing_loader, csv_filename, print_results=False):
    model.eval()

    # Get the current directory
    current_directory = os.getcwd()

    # Create a new folder for saving the images
    save_directory = os.path.join(current_directory, 'image_detections')
    os.makedirs(save_directory, exist_ok=True)

    # Evaluate the model
    with torch.no_grad():
        category_correct_boxes = defaultdict(int)
        category_correct_labels = defaultdict(int)
        category_total_boxes = defaultdict(int)

        for imgs, annotations in testing_loader:
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            i = 0
            for idx, image in enumerate(imgs):
                i += 1
                
                image_np = image.cpu().numpy().transpose(1, 2, 0)

                boxes, pred_classes, labels = detect_utils.predict(image_np, model, device, 0.4)
                # print("labels: ")
                # print(pred_classes)

                gt_boxes = annotations[idx]['boxes'].cpu().numpy()
                gt_labels = annotations[idx]['labels'].cpu().numpy()

                correct_labels = []

                for label_index in gt_labels:
                    if label_index == 32:
                        label_index = 0
                    correct_labels.append(COCO_INSTANCE_CATEGORY_NAMES[label_index])

                # print(correct_labels)
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

                    # Save the image with bounding boxes
                    image_with_boxes = draw_boxes(image, boxes, gt_label, i)
                    save_path = os.path.join(save_directory, f'image_{idx}.jpg')
                    # cv2.imwrite(save_path, image_with_boxes)

        # Write results to CSV file
        with open(csv_filename, 'w', newline='') as csvfile:
            fieldnames = ['category', 'bounding_box_accuracy', 'label_accuracy']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for category in category_total_boxes.keys():
                box_accuracy = category_correct_boxes[category] / category_total_boxes[category]
                label_accuracy = category_correct_labels[category] / category_total_boxes[category]

                writer.writerow({'category': category, 'bounding_box_accuracy': box_accuracy, 'label_accuracy': label_accuracy})
        
        # Write results to Excel file
        results = []
        for category in category_total_boxes.keys():
            box_accuracy = category_correct_boxes[category] / category_total_boxes[category]
            label_accuracy = category_correct_labels[category] / category_total_boxes[category]
            results.append({'category': category, 'bounding_box_accuracy': box_accuracy, 'label_accuracy': label_accuracy})

        excel_filename = 'results.xlsx'
        df = pd.DataFrame(results)
        df.to_excel(excel_filename, index=False)

        
        # Calculate and print accuracy for each category
        for category in category_total_boxes.keys():
            box_accuracy = category_correct_boxes[category] / category_total_boxes[category]
            label_accuracy = category_correct_labels[category] / category_total_boxes[category]

            print(f"Category: {category}")
            print(f"Bounding Box Accuracy: {box_accuracy * 100:.2f}%")
            print(f"Label Accuracy: {label_accuracy * 100:.2f}%")

def draw_boxes(image_tensor, boxes, label, i):
    # Convert image tensor to numpy array
    pil_image = transforms.ToPILImage()(image_tensor)
    pil_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    try:
        for box, lbl in zip(boxes, label):
            x1, y1, x2, y2 = box
            cv2.rectangle(pil_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Add label text above the box
            cv2.putText(pil_image, lbl, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Save image with bounding boxes
        save_path = os.path.join(os.getcwd(), 'detections/' + str(i) + '.jpg')
        cv2.imwrite(save_path, pil_image)
    except Exception as e:
        print("Error:", str(e))

    return pil_image


