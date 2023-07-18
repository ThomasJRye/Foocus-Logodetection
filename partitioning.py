import json
import random

# Load COCO annotations file
with open("Thomas_Foocus_COCO/instances_default.json", "r") as f:
    coco = json.load(f)

# Create empty lists for train and test annotations
train = {"images": [], "annotations": [], "categories": coco["categories"]}
test = {"images": [], "annotations": [], "categories": coco["categories"]}

# Define split ratio
split_ratio = 0.8 # 80% train, 20% test

# Loop over images and annotations and randomly assign them to train or test
for image in coco["images"]:
    # Get image id
    image_id = image["id"]

    # Get corresponding annotations
    anns = [ann for ann in coco["annotations"] if ann["image_id"] == image_id]

    # Randomly choose train or test based on split ratio
    if random.random() < split_ratio:
        # Assign image and annotations to train
        train["images"].append(image)
        train["annotations"].extend(anns)
    else:
        # Assign image and annotations to test
        test["images"].append(image)
        test["annotations"].extend(anns)

# Save train and test annotations as new JSON files
with open("Thomas_Foocus_COCO/train.json", "w") as f:
    json.dump(train, f)

with open("Thomas_Foocus_COCO/test.json", "w") as f:
    json.dump(test, f)