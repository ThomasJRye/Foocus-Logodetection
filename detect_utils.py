import torchvision.transforms as transforms
import cv2
import numpy as np
import torch
from coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names

np.random.seed(42)

# Create different colors for each class.
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

# Define the torchvision image transforms.
transform = transforms.Compose([
    transforms.ToTensor(),
])

def predict(image, model, device, detection_threshold):
    """
    Predict the output of an image after forward pass through
    the model and return the bounding boxes, class names, and 
    class labels. 
    """
    
    # Transform the image to tensor.
    image = transform(image).to(device)
    # Add a batch dimension.
    image = image.unsqueeze(0) 
    # Get the predictions on the image.
    with torch.no_grad():
        outputs = model(image) 
    # Get score for all the predicted objects.
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    # Get all the predicted bounding boxes.
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    # Get boxes above the threshold score.
    boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)
    labels = outputs[0]['labels'][:len(boxes)]


    label_indexes = labels.cpu().numpy() - 1
    # Get all the predicited class names.
    # pred_classes = [coco_names[i] if 0 <= i < len(coco_names) else 'unknown' for i in label_indexes]

    pred_classes = [coco_names[i] if 0 <= i < len(coco_names) else 'unknown' for i in labels.cpu().numpy()]

    return boxes, pred_classes, labels


def draw_boxes(boxes, classes, labels, image):
    # read the image with OpenCV
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        cv2.putText(image, classes[i], (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, 
                    lineType=cv2.LINE_AA)
    return image

def calculate_iou(box1, box2):
    """
    Calculate the Intersection Over Union (IOU) of two bounding boxes.
    Args:
        box1: A list or array of the format [x1, y1, x2, y2]
        box2: A list or array of the format [x1, y1, x2, y2]
    Returns:
        iou: A float representing the IOU value.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = box1_area + box2_area - intersection
    iou = intersection / union if union != 0 else 0
    return iou
