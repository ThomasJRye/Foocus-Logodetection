import json
import os

def main():
    coco_annotation_file_path = "bigData2/new_new_labels.json"

    coco = json.load(open(coco_annotation_file_path))
    new_coco = coco
    for ann in coco['annotations']:
        if ann['bbox'] == [146.5, 356.2, 42.33, 16.5]:
            new_coco['annotations'].remove(ann)
            print(ann)

    with open('bigData2/new_new_labels.json', 'w') as outfile:
        json.dump(new_coco, outfile)
    
def count():
    coco_annotation_file_path = "bigData2/new_new_labels.json"

    coco = json.load(open(coco_annotation_file_path))
    i=0
    for ann in coco['annotations']:
        i+=1
    print(i)
    
    coco_annotation_file_path = "bigData2/labels.json"

    coco = json.load(open(coco_annotation_file_path))
    i=0
    for ann in coco['annotations']:
        i+=1
    print(i)
if __name__ == "__main__":
    count()