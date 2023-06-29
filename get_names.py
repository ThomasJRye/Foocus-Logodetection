import json

# Specify the path to your COCO annotations JSON file
coco_annotations_file = '/Users/thomasrye/Documents/github/Foocus-Logodetection/bigData2/labels_coop.json'

# Open the JSON file
with open(coco_annotations_file, 'r') as file:
    coco_annotations = json.load(file)

# Extract the category names
category_names = [category['name'] for category in coco_annotations['categories']]

# Print the category names
for name in category_names:
    print(name)

