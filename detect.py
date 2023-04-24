import torch
import torchvision
import argparse
import cv2
import detect_utils
import numpy as np
from PIL import Image
from model import get_model

# Construct the argument parser.
parser = argparse.ArgumentParser()

parser.add_argument(
    '-i', '--input', default='input/image_1.jpg', 
    help='path to input input image'
)
parser.add_argument('-m', '--min-size', dest='min_size', default=800, 
                    help='minimum input size for the FasterRCNN network')
parser.add_argument(
    '-t', '--threshold', default=0.5, type=float,
    help='detection threshold'
)
parser.add_argument(
    '--model', default='v2', 
    help='faster rcnn resnet50 fpn or fpn v2',
    choices=['v1', 'v2']
)

args = vars(parser.parse_args())

# Define the computation device.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = get_model()

model.load_state_dict(torch.load('models/bigdata.pth', map_location=device))

# # download or load the model from disk
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, 
#                                                     min_size=args['min_size'])
image = Image.open(args['input'])

model.eval().to(device)

boxes, classes, labels = detect_utils.predict(image, model, device, 0.8)

image = detect_utils.draw_boxes(boxes, classes, labels, image)

cv2.imshow('Image', image)
save_name = f"{args['input'].split('/')[-1].split('.')[0]}_{args['min_size']}"
cv2.imwrite(f"outputs/{save_name}.jpg", image)
cv2.waitKey(0)