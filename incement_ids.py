import os
import json
import shutil
cwd = os.getcwd()
im_directory = '/Users/thomasrye/Documents/github/Foocus-Logodetection/bigData2/data/data 2/' 
# Get a list of all files in the directory
files = os.listdir(im_directory)


# # Iterate over the files
# for file in files:
#     if file == '.DS_Store':
#         continue
#     old_name = os.path.join(im_directory, file)
#     print(file)
#     file_id = int(file.rstrip(".jpg"))
#     new_id = file_id+74
    

#     new_name = os.path.join('/Users/thomasrye/Documents/github/Foocus-Logodetection/images', str(int(new_id)) + ".jpg")
    
#     print(old_name)
#     print(new_id)
#     # Rename the file
#     shutil.copy(im_directory + file, new_name)
    
coco_annotation_file_path = cwd + "/bigData2/labels_with_extra.json"

coco = json.load(open(coco_annotation_file_path))
new_coco = {}
new_coco['images'] = []
new_coco['annotations'] = []

image_ids = []
for image in coco['images']:
    file_name = image['file_name']

    if 'images' in file_name:
        # save image id to find annotations later
        image_ids.append(image['id'])

        # only get number from file name
        numeric_text = ''.join([char for char in file_name if char.isdigit()])
        name_id = int(numeric_text)
        new_name_id = name_id + 74
        new_name = 'images/' + str(new_name_id) + '.jpg'
        image['file_name'] = new_name
        image['id'] = image['id'] + 74
        new_coco['images'].append(image)
    
for ann in coco['annotations']:
    if ann['image_id'] in image_ids:
        ann['image_id'] = ann['image_id'] + 73
        new_coco['annotations'].append(ann)
        

# Save the updated COCO annotation file
with open(cwd + "/bigData2/labels_with_extra_new.json", 'w') as outfile:
    json.dump(new_coco, outfile)
