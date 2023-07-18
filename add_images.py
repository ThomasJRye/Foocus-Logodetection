import os
import json
from PIL import Image
from pycocotools.coco import COCO

# Path to the existing COCO dataset
coco_dir = "/Users/thomasrye/Documents/github/Foocus-Logodetection/bigData2/"

# Path to the folder containing new images
new_images_dir = os.path.join(coco_dir, "data/")

# Load the COCO annotation file
annotation_file = os.path.join(coco_dir, "labels.json")
coco = COCO(annotation_file)

# Get the list of existing image file names in the COCO dataset
existing_images = set([image['file_name'] for image in coco.dataset['images']])

# Get the list of new image file names
new_images = [filename for filename in os.listdir(new_images_dir) if filename.endswith(('.jpg', '.jpeg', '.png'))]

i = 2426
# Loop through the new images and add them to the COCO dataset if they're not already present
for filename in new_images:
    if filename not in existing_images:
        i += 1
        # Add image to the dataset
        image_id = i
        image_path = os.path.join(new_images_dir, filename)
        image = Image.open(image_path)
        width, height = image.size

        coco.dataset['images'].append({
            'id': image_id,
            'file_name': filename,
            'width': width,
            'height': height
        })

# Save the updated COCO annotation file
new_annotation_file = os.path.join(coco_dir, "instances_train2017_new.json")
with open(new_annotation_file, 'w') as f:
    json.dump(coco.dataset, f)
